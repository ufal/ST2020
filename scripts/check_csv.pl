#!/usr/bin/env perl
# Reads a WALS CSV file from the SIGTYP shared task. Tests it for consistency (such as fixed number of columns, because some feature values may contain tabulators).
# Copyright © 2020 Dan Zeman <zeman@ufal.mff.cuni.cz>
# License: GNU GPL

use utf8;
use open ':utf8';
binmode(STDIN, ':utf8');
binmode(STDOUT, ':utf8');
binmode(STDERR, ':utf8');
use Getopt::Long;

my $original = 0; # original data separated by spaces and tabs, with errors; our _x and _y data (original==0) separated by commas
my $rcomma = "\x{3001}"; # 、 to replace comma inside of quotation marks for easier processing
GetOptions
(
    'original' => \$original
);

my @headers = ();
my $nf;
my $iline = 0;
my @data; # the table, without the header line
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
            print STDERR ("Line $iline ($f[1]): Expected $nf fields, found $n.\n");
        }
        push(@data, \@f);
    }
}
###!!! While I originally targeted both file formats, the code from here on assumes the order of columns of the non-original file.
if($original)
{
    die("Cannot do this with the original file.");
}
# Hash the observed features and values.
my %h;
my %lh; # hash indexed by language code
foreach my $language (@data)
{
    my $lcode = $language->[1];
    # Remember observed features and values.
    for(my $i = 1; $i <= $#headers; $i++)
    {
        my $feature = $headers[$i];
        # There seem to be multiple ways of indicating an unknown value. Unify them.
        $language->[$i] = '?' if(!defined($language->[$i]) || $language->[$i] eq '' || $language->[$i] eq 'nan' || $language->[$i] eq '?');
        $h{$feature}{$language->[$i]}++;
        $lh{$lcode}{$feature} = $language->[$i];
    }
}
#list_features_and_values(\@headers, \%h);
compute_pairwise_cooccurrence(\@headers, \%lh);



#------------------------------------------------------------------------------
# Prints statistics of features and their values.
#------------------------------------------------------------------------------
sub list_features_and_values
{
    my $headers = shift; # array ref
    my $h = shift; # hash ref
    for(my $i = 3; $i <= $#{$headers}; $i++)
    {
        print("$headers->[$i]\n\n");
        my @values = sort {my $r = $h->{$headers->[$i]}{$b} <=> $h->{$headers->[$i]}{$a}; unless($r) {$r = $a cmp $b} $r} (keys(%{$h->{$headers->[$i]}}));
        # Compute probabilities of known values.
        my $sum = 0;
        foreach my $value (@values)
        {
            unless($value eq '?')
            {
                $sum += $h->{$headers->[$i]}{$value}
            }
        }
        foreach my $value (@values)
        {
            my $p = '';
            unless($value eq '?')
            {
                $p = $h->{$headers->[$i]}{$value} / $sum;
            }
            print("  $value\t$h->{$headers->[$i]}{$value}\t$p\n");
        }
        print("\n");
    }
}



#------------------------------------------------------------------------------
# Computes pairwise cooccurrence of two feature values in a language.
#------------------------------------------------------------------------------
sub compute_pairwise_cooccurrence
{
    my $headers = shift; # array ref
    my $lh = shift; # hash ref
    my @languages = keys(%{$lh});
    my %cooc;
    foreach my $l (@languages)
    {
        foreach my $f (@{$headers})
        {
            next if(!exists($lh->{$l}{$f}));
            my $fv = $lh->{$l}{$f};
            next if($fv eq '?');
            foreach my $g (@{$headers})
            {
                next if($g eq $f);
                next if(!exists($lh->{$l}{$g}));
                my $gv = $lh->{$l}{$g};
                next if($gv eq '?');
                $cooc{$f}{$fv}{$g}{$gv}++;
                ###!!! DEBUG
                #print("$l\n");
                #print("$f = $fv\n");
                #print("$g = $gv\n");
                #print("\n");
            }
        }
    }
    # Now look at the cooccurrences disregarding individual languages.
    my %coocflat;
    foreach my $f (@{$headers})
    {
        my @fvalues = keys(%{$cooc{$f}});
        foreach my $fv (@fvalues)
        {
            foreach my $g (keys(%{$cooc{$f}{$fv}}))
            {
                # Cooccurrences are symmetric, hence do not report both $f+$g and $g+$f.
                next if($g le $f);
                my @gvalues = keys(%{$cooc{$f}{$fv}{$g}});
                foreach my $gv (@gvalues)
                {
                    my $keystring = "$f == $fv && $g == $gv";
                    $coocflat{$keystring} = $cooc{$f}{$fv}{$g}{$gv};
                }
            }
        }
    }
    my @keys = sort {$coocflat{$b} <=> $coocflat{$a}} (keys(%coocflat));
    foreach my $key (@keys)
    {
        last if($coocflat{$key} <= 5);
        print("$coocflat{$key}\t$key\n");
    }
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
