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
# Takes the column headers (needed because of their order) and the current
# language-feature-score hash with the scores of the newly predicted values
# and prints the scores as a CSV file to STDOUT.
#------------------------------------------------------------------------------
sub write_scores
{
    my $data = shift; # hash ref
    my $filename = shift;
    open(FILE, ">$filename") or confess("Cannot write '$filename': $!");
    my @headers = map {escape_commas($_)} (@{$data->{infeatures}});
    print FILE (join(',', @headers), "\n");
    my @lcodes = sort {$data->{lh}{$a}{index} <=> $data->{lh}{$b}{index}} (@{$data->{lcodes}});
    foreach my $l (@lcodes)
    {
        my @values = map
        {
            my $v = exists($data->{scores}{$l}{$_}) ? $data->{scores}{$l}{$_} : 'inf';
            escape_commas($v);
        }
        (@{$data->{infeatures}});
        print FILE (join(',', @values), "\n");
    }
    close(FILE);
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
        $_ = process_quoted_commas($_, $rcomma);
        my @f = ();
        if(scalar(@headers)==0)
        {
            @f = split(/,/, $_);
            @headers = map {restore_commas($_, $rcomma)} (@f);
            $nf = scalar(@headers);
        }
        else
        {
            @f = split(/,/, $_);
            @f = map {restore_commas($_, $rcomma)} (@f);
            my $n = scalar(@f);
            # The number of values may be less than the number of columns if the trailing columns are empty.
            # However, the number of values must not be greater than the number of columns (which would happen if a value contained the separator character).
            if($n > $nf)
            {
                # Assume that it can be fixed by joining the last column with the extra columns.
                # The tabulator seems to appear at the boundary of two features, and there is no vertical bar, so we should probably add '|'.
                my $ntojoin = $n-$nf+1;
                print STDERR ("Line $iline ($f[1]): Expected $nf fields, found $n. Joining the last $ntojoin fields.\n");
                $f[$nf-1] = join('|', @f[($nf-1)..($n-1)]);
                splice(@f, $nf);
            }
            push(@data, \@f);
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
# Reads the input CSV (comma-separated values) file: either from STDIN, or from
# files supplied as arguments. Returns a list of column headers and a list of
# table rows (both as array refs).
#
# This is a new implementation that tries to be more general (and in particular
# suitable for reading WALS). Quoted values may contain line breaks, which
# means that we cannot rely on processing the input one line at a time.
#------------------------------------------------------------------------------
sub read_csv1
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
    my $iline = 0;
    my $state = 'rowbegin'; # rowbegin | cellbegin | inquotes | cellend
    my $buffer = '';
    my @f = ();
    my @data;
    while(<>)
    {
        $iline++;
        s/\r?\n$//;
        my $line = $_;
        while($line)
        {
            if($state =~ m/^(rowbegin|cellbegin)$/)
            {
                if($line =~ s/^"//) # "
                {
                    $state = 'inquotes';
                }
                else
                {
                    # Unquoted table cell. Read everything until the next comma or line end.
                    if($line =~ s/^([^,]*)//)
                    {
                        $buffer = $1;
                    }
                    push(@f, $buffer);
                    $buffer = '';
                    $state = 'cellend';
                }
            }
            elsif($state eq 'cellend')
            {
                if($line =~ s/^,//)
                {
                    $state = 'cellbegin';
                }
                elsif($line ne '')
                {
                    print STDERR ("WARNING: Cell has ended, no comma found, ignoring the rest of the line: '$line'\n");
                }
                else # nothing more on the line, and the last cell has been terminated
                {
                    my @row = @f;
                    push(@data, \@row);
                    @f = ();
                    $state = 'rowbegin';
                }
            }
            elsif($state eq 'inquotes')
            {
                # Double quotation mark serves as an escape for quotation mark.
                if($line =~ s/^""//)
                {
                    $buffer .= '"';
                }
                elsif($line =~ s/^([^"]+)//) # "
                {
                    $buffer .= $1;
                    # If we consumed the rest of the line now, the current cell will continue on the next line.
                    if($line eq '')
                    {
                        $buffer .= "\n";
                    }
                }
                elsif($line =~ s/^"//) # "
                {
                    push(@f, $buffer);
                    $buffer = '';
                    $state = 'cellend';
                }
            }
        }
    }
    if($state ne 'rowbegin')
    {
        print STDERR ("WARNING: Reading the input ended in an unexpected state '$state'\n");
    }
    # The first row contains the headers.
    my $headers = shift(@data);
    if($myfiles)
    {
        @ARGV = @oldargv;
    }
    ###!!! The remaining adjustments are here because of compatibility with our earlier
    ###!!! read_csv() function. They are specific to the SIGTYP shared task and they
    ###!!! should not be in a general CSV reading function.
    if(scalar(@headers) > 0 && $headers[0] eq '')
    {
        $headers[0] = 'index';
    }
    # We may want to add new combined features but we will not want to output them;
    # therefore we must save the original list of features.
    my @infeatures = @{$headers};
    my %data =
    (
        'infeatures' => \@infeatures,
        'features'   => $headers,
        'table'      => \@data,
        'nf'         => scalar(@infeatures),
        'nl'         => scalar(@data)
    );
    return %data;
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



#------------------------------------------------------------------------------
# Reads a table as a tab-separated text file: either from STDIN, or from files
# supplied as arguments. Expects certain anomalies that appear in the data
# provided by the shared task organizers: the last column sometimes contains
# a tab character which is not meant to start a new column.
#------------------------------------------------------------------------------
sub read_tab
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
    my $id = 0;
    my %feature_names;
    while(<>)
    {
        $iline++;
        s/\r?\n$//;
        if(scalar(@headers)==0)
        {
            # In the original dev.csv, the headers are separated by one or more spaces, while the rest is separated by tabulators!
            @headers = split(/\s+/, $_);
            # Insert the column for the numeric id that we use in our format.
            unshift(@headers, 'index');
            $nf = scalar(@headers);
        }
        else
        {
            my @f = split(/\t/, $_);
            # Insert the column for the numeric id that we use in our format.
            unshift(@f, $id++);
            my $n = scalar(@f);
            # The number of values may be less than the number of columns if the trailing columns are empty.
            # However, the number of values must not be greater than the number of columns (which would happen if a value contained the separator character).
            if($n > $nf)
            {
                # Assume that it can be fixed by joining the last column with the extra columns.
                # The tabulator seems to appear at the boundary of two features, and there is no vertical bar, so we should probably add '|'.
                ###!!! WARNING! In fact, the shared task organizers made different flavors of this error in train+dev and in test_blinded data.
                ###!!! The train and dev sets do contain the vertical bar (and it is preceded by a rest of the value of the feature that contains TAB).
                ###!!! The test_blinded data omit the rest of the value and the vertical bar.
                my $ntojoin = $n-$nf+1;
                print STDERR ("Line $iline ($f[1]): Expected $nf fields, found $n. Joining the last $ntojoin fields.\n");
                if($ntojoin == 2)
                {
                    print STDERR ("Focused joining strategy.\n");
                    if($f[$nf-1] =~ m/Verb-Initial_with_Preverbal_Negative=1 Separate word, no double negation$/ &&
                       $f[$nf] !~ m/^Word\&NoDoubleNeg/)
                    {
                        $f[$nf-1] .= "\tWord\&NoDoubleNeg";
                    }
                    if($f[$nf-1] =~ m/Verb-Initial_with_Preverbal_Negative=2 Prefix, no double negation$/ &&
                       $f[$nf] !~ m/^Prefix\&NoDoubleNeg/)
                    {
                        $f[$nf-1] .= "\tPrefix\&NoDoubleNeg";
                    }
                    $f[$nf-1] = join('|', @f[($nf-1)..$nf]);
                    splice(@f, $nf);
                }
                else
                {
                    print STDERR ("Default joining strategy.\n");
                    $f[$nf-1] = join("\t", @f[($nf-1)..($n-1)]);
                    splice(@f, $nf);
                }
            }
            # The original format has 8 columns, now 9 because we added the numeric index.
            # The ninth column contains all the features. Remember their names.
            if(scalar(@f)==9)
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
    # Split the features so that each has its own column.
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
    if($myfiles)
    {
        @ARGV = @oldargv;
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
# Converts the input table to a hash. First part, no further sophisticated
# indexing. The idea is that from now on, we will use {lh} instead of {table}.
# Therefore we remove {table} at the end (nobody will update it if {lh}
# changes). On the other hand, {features} is still useful and we keep it.
# INPUT:
#   {features} .... list of feature names
#   {table} ....... table of feature values for each language
# OUTPUT:
#   {lcodes} ...... list of wals codes of known languages (index to lh)
#   {lh} .......... data from {table} indexed by {language}{feature}
#------------------------------------------------------------------------------
sub convert_table_to_lh
{
    my $data = shift; # hash ref
    my $qm_is_nan = shift; # convert question marks to 'nan'?
    my @lcodes;
    my %lh; # hash indexed by language code
    # Create the initial language-feature-value hash but do not infer anything
    # further yet. We may want to modify some features before we proceed.
    foreach my $line (@{$data->{table}})
    {
        my $lcode = $line->[1];
        push(@lcodes, $lcode);
        for(my $i = 0; $i <= $#{$data->{features}}; $i++)
        {
            my $feature = $data->{features}[$i];
            # Make sure that a feature missing from the database is always indicated as 'nan' (normally 'nan' appears already in the input).
            $line->[$i] = 'nan' if(!defined($line->[$i]) || $line->[$i] eq '' || $line->[$i] eq 'nan');
            # Our convention: a question mark masks a feature value that is available in WALS but we want our model to predict it.
            # If desired, we can convert question marks to 'nan' here.
            $line->[$i] = 'nan' if($line->[$i] eq '?' && $qm_is_nan);
            $lh{$lcode}{$feature} = $line->[$i];
        }
    }
    $data->{lcodes} = \@lcodes;
    $data->{lh} = \%lh;
    delete($data->{table});
}



#------------------------------------------------------------------------------
# Reads WALS data in the new format (set of CSV files), released March 2020.
#------------------------------------------------------------------------------
sub read_wals
{
    my $path = shift; # path to the folder with all the necessary CSV files: e.g. 'data/wals-2020/cldf'
    # Read the languages and their core features (attributes).
    my %languages_in = read_csv1("$path/languages.csv");
    # Hash the languages by their ids (WALS codes).
    if(join(',', @{$languages_in{infeatures}}) ne 'ID,Name,Macroarea,Latitude,Longitude,Glottocode,ISO639P3code,Family,Subfamily,Genus,ISO_codes,Samples_100,Samples_200')
    {
        die("Unexpected headers in languages.csv");
    }
    my %languages;
    foreach my $l (@{$languages_in{table}})
    {
        my %record =
        (
            'id'           => $l->[0],
            'name'         => $l->[1],
            'macroarea'    => $l->[2],
            'latitude'     => $l->[3],
            'longitude'    => $l->[4],
            'glottocode'   => $l->[5],
            'iso639p3code' => $l->[6],
            'family'       => $l->[7],
            'subfamily'    => $l->[8],
            'genus'        => $l->[9],
            'iso_codes'    => $l->[10],
            'samples_100'  => $l->[11],
            'samples_200'  => $l->[12]
        );
        $languages{$record{id}} = \%record;
    }
    # Read the feature ("parameter") names and codes.
    my %parameters_in = read_csv1("$path/parameters.csv");
    # Hash the parameters by their ids.
    if(join(',', @{$parameters_in{infeatures}}) ne 'ID,Name,Description,Contributor_ID,Chapter,Area')
    {
        die("Unexpected headers in parameters.csv");
    }
    my %parameters;
    foreach my $p (@{$parameters_in{table}})
    {
        my %record =
        (
            'id'             => $p->[0],
            'name'           => $p->[1],
            'description'    => $p->[2],
            'contributor_id' => $p->[3],
            'chapter'        => $p->[4],
            'area'           => $p->[5]
        );
        $parameters{$record{id}} = \%record;
    }
    # Read the feature value repertory ("codes").
    my %codes_in = read_csv1("$path/codes.csv");
    # Hash the feature values by their ids.
    if(join(',', @{$codes_in{infeatures}}) ne 'ID,Parameter_ID,Name,Description,Number,icon')
    {
        die("Unexpected headers in codes.csv");
    }
    my %codes;
    foreach my $c (@{$codes_in{table}})
    {
        my %record =
        (
            'id'           => $c->[0],
            'parameter_id' => $c->[1],
            'name'         => $c->[2],
            'description'  => $c->[3],
            'number'       => $c->[4],
            'icon'         => $c->[5]
        );
        $codes{$record{id}} = \%record;
    }
    # Read the values of the features of individual languages.
    my %values_in = read_csv1("$path/values.csv");
    # Hash the feature values by their ids.
    if(join(',', @{$values_in{infeatures}}) ne 'ID,Language_ID,Parameter_ID,Value,Code_ID,Comment,Source,Example_ID')
    {
        die("Unexpected headers in values.csv");
    }
    my %values;
    foreach my $v (@{$values_in{table}})
    {
        my %record =
        (
            'id'           => $v->[0],
            'language_id'  => $v->[1],
            'parameter_id' => $v->[2],
            'value'        => $v->[3],
            'code_id'      => $v->[4],
            'comment'      => $v->[5],
            'source'       => $v->[6],
            'example_id'   => $v->[7]
        );
        $values{$record{id}} = \%record;
    }
    # Package all the hashes in one and return it.
    my %wals =
    (
        'languages'  => \%languages,
        'parameters' => \%parameters,
        'codes'      => \%codes,
        'values'     => \%values
    );
    return %wals;
}



1;
