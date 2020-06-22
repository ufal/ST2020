package Sigtypio;

use utf8;
use open ':utf8';
use namespace::autoclean;

use Carp;



#------------------------------------------------------------------------------
# Takes the column headers (needed because of their order) and the current
# language-feature hash with the newly predicted values and prints them as
# a CSV file to STDOUT.
#------------------------------------------------------------------------------
sub write_csv
{
    my $data = shift; # hash ref
    my $filename = shift; # optional path to the output file; if not present, we will write to STDOUT
    if(defined($filename))
    {
        open(FILE, ">$filename") or confess("Cannot write '$filename': $!");
    }
    my @headers = map {escape_commas($_)} (@{$data->{infeatures}});
    if(defined($filename))
    {
        print FILE (join(',', @headers), "\n");
    }
    else
    {
        print(join(',', @headers), "\n");
    }
    my @lcodes = sort {$data->{lh}{$a}{index} <=> $data->{lh}{$b}{index}} (@{$data->{lcodes}});
    foreach my $l (@lcodes)
    {
        my @values = map
        {
            # If we modified some values of some input features, restore the
            # original values now.
            my $v = exists($data->{restore}{$l}{$_}) ? $data->{restore}{$l}{$_} : $data->{lh}{$l}{$_};
            escape_commas($v);
        }
        (@{$data->{infeatures}});
        if(defined($filename))
        {
            print FILE (join(',', @values), "\n");
        }
        else
        {
            print(join(',', @values), "\n");
        }
    }
    if(defined($filename))
    {
        close(FILE);
    }
}



#------------------------------------------------------------------------------
# Takes a value of a CSV cell and makes sure that it is enclosed in quotation
# marks if it contains a comma.
#------------------------------------------------------------------------------
sub escape_commas
{
    my $string = shift;
    if($string =~ m/"/) # "
    {
        die("A CSV value must not contain a double quotation mark");
    }
    if($string =~ m/,/)
    {
        $string = '"'.$string.'"';
    }
    return $string;
}



#------------------------------------------------------------------------------
# Reads the input CSV (comma-separated values) file: either from STDIN, or from
# files supplied as arguments. Returns a list of column headers and a list of
# table rows (both as array refs).
#------------------------------------------------------------------------------
sub read_csv
{
    # @_ can optionally contain the list of files to read from.
    # If it does not exist, @ARGV will be tried. If it is empty, read from STDIN.
    my $myfiles = 0;
    my @oldargv;
    if(scalar(@_) > 0)
    {
        $myfiles = 1;
        @oldargv = @ARGV;
        @ARGV = @_;
    }
    my @headers = ();
    my $nf;
    my $iline = 0;
    my @data; # the table, without the header line
    my $rcomma = chr(12289); # IDEOGRAPHIC COMMA
    my $id = 0;
    my %feature_names;
    while(<>)
    {
        $iline++;
        s/\r?\n$//;
        # In the non-original (comma-separated) file, we must distinguish commas inside quotes from those outside.
        unless($original)
        {
            $_ = process_quoted_commas($_, $rcomma);
        }
        my @f = ();
        if(scalar(@headers)==0)
        {
            # In the original dev.csv, the headers are separated by one or more spaces, while the rest is separated by tabulators!
            if($original)
            {
                @f = split(/\s+/, $_);
                # Insert the column for the numeric id that we use in our format.
                unshift(@f, '');
            }
            else
            {
                @f = split(/,/, $_);
            }
            @headers = map {restore_commas($_, $rcomma)} (@f);
            $nf = scalar(@headers);
        }
        else
        {
            if($original)
            {
                @f = split(/\t/, $_);
                # Insert the column for the numeric id that we use in our format.
                unshift(@f, $id++);
            }
            else
            {
                @f = split(/,/, $_);
            }
            @f = map {restore_commas($_, $rcomma)} (@f);
            my $n = scalar(@f);
            # The number of values may be less than the number of columns if the trailing columns are empty.
            # However, the number of values must not be greater than the number of columns (which would happen if a value contained the separator character).
            if($n > $nf)
            {
                # Assume that it can be fixed by joining the last column with the extra columns.
                my $ntojoin = $n-$nf+1;
                print STDERR ("Line $iline ($f[1]): Expected $nf fields, found $n. Joining the last $ntojoin fields.\n");
                $f[$nf-1] = join('', @f[($nf-1)..($n-1)]);
                splice(@f, $nf);
            }
            # The original format has 8 columns, now 9 because we added the numeric index.
            # The ninth column contains all the features. Remember their names.
            if($original && scalar(@f)==9)
            {
                my @features = map {s/=.*//; $_} (split(/\|/, $f[8]));
                foreach my $feature (@features)
                {
                    $feature_names{$feature}++;
                }
            }
            push(@data, \@f);
        }
    }
    # If we read the original format, split the features so that each has its own column.
    if($original)
    {
        my @fnames = sort(keys(%feature_names));
        splice(@headers, $#headers, 1, @fnames);
        foreach my $row (@data)
        {
            my @features = split(/\|/, pop(@{$row}));
            my %features;
            foreach my $fv (@features)
            {
                if($fv =~ m/^(.+?)=(.+)$/)
                {
                    $features{$1} = $2;
                }
            }
            foreach my $f (@fnames)
            {
                if(defined($features{$f}))
                {
                    push(@{$row}, $features{$f});
                }
                else
                {
                    push(@{$row}, 'nan');
                }
            }
        }
    }
    if($myfiles)
    {
        @ARGV = @oldargv;
    }
    if(scalar(@headers) > 0 && $headers[0] eq '')
    {
        $headers[0] = 'index';
    }
    # We may want to add new combined features but we will not want to output them;
    # therefore we must save the original list of features.
    my @infeatures = @headers;
    my %data =
    (
        'infeatures' => \@infeatures,
        'features'   => \@headers,
        'table'      => \@data,
        'nf'         => scalar(@headers),
        'nl'         => scalar(@data)
    );
    return %data;
}



#------------------------------------------------------------------------------
# Substitutes commas surrounded by quotation marks. Removes quotation marks.
#------------------------------------------------------------------------------
sub process_quoted_commas
{
    my $string = shift;
    my $rcomma = shift; # the replacement
    # We will replace inside commas by another character.
    # Perform a sanity check first: the replacement character must not appear in the input.
    if($string =~ m/$rcomma/)
    {
        die("Want to use '$rcomma' as a replacement for comma but it occurs in the input.");
    }
    my @inchars = split(//, $string);
    my @outchars = ();
    my $inside = 0;
    foreach my $char (@inchars)
    {
        # Assume that the quotation mark always has the special function and never occurs as a part of the value.
        if($char eq '"')
        {
            if($inside)
            {
                $inside = 0;
            }
            else
            {
                $inside = 1;
            }
        }
        else
        {
            if($inside && $char eq ',')
            {
                push(@outchars, $rcomma);
            }
            else
            {
                push(@outchars, $char);
            }
        }
    }
    $string = join('', @outchars);
    return $string;
}



#------------------------------------------------------------------------------
# Once the input line has been split to fields, the commas in the fields can be
# restored.
#------------------------------------------------------------------------------
sub restore_commas
{
    my $string = shift;
    my $rcomma = shift; # the replacement
    $string =~ s/$rcomma/,/g;
    return $string;
}



1;
