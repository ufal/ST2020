#!/usr/bin/env perl
# Reads the training data train_y.csv and computes conditional probabilities of language features.
# Reads the blind development data dev_x.csv, replaces question marks by predicted features and writes the resulting file.
# Copyright © 2020 Dan Zeman <zeman@ufal.mff.cuni.cz>
# License: GNU GPL

use utf8;
use open ':utf8';
binmode(STDIN, ':utf8');
binmode(STDOUT, ':utf8');
binmode(STDERR, ':utf8');



my $data_folder = 'data';
print STDERR ("Reading the training data...\n");
my ($trainheaders, $traindata) = read_csv("$data_folder/train_y.csv");
print STDERR ("Found ", scalar(@{$trainheaders}), " headers.\n");
print STDERR ("Found ", scalar(@{$traindata}), " language lines.\n");
print STDERR ("Hashing the features and their cooccurrences...\n");
# Hash the observed features and values.
my ($trainh, $trainlh) = hash_features($trainheaders, $traindata, 0);
my ($traincooc, $trainprob) = compute_pairwise_cooccurrence($trainheaders, $trainlh);
print STDERR ("Reading the development data...\n");
my ($devheaders, $devdata) = read_csv("$data_folder/dev_x.csv");
print STDERR ("Found ", scalar(@{$devheaders}), " headers.\n");
print STDERR ("Found ", scalar(@{$devdata}), " language lines.\n");
my $ndevlangs = scalar(@{$devdata});
my $ndevfeats = scalar(@{$devheaders})-1; # first column is ord number; except for that, counting everything including the language code and name
my $ndevlangfeats = $ndevlangs*$ndevfeats;
print STDERR ("$ndevlangs languages × $ndevfeats features would be $ndevlangfeats.\n");
my ($devh, $devlh) = hash_features($devheaders, $devdata, 0);
my @features = keys(%{$devh});
my $nnan = 0;
my $nqm = 0;
my $nreg = 0;
foreach my $f (@features)
{
    my @values = keys(%{$devh->{$f}});
    foreach my $v (@values)
    {
        if($v eq 'nan')
        {
            $nnan += $devh->{$f}{$v};
        }
        elsif($v eq '?')
        {
            $nqm += $devh->{$f}{$v};
        }
        else
        {
            $nreg += $devh->{$f}{$v};
        }
    }
}
print STDERR ("Found $nnan nan values.\n");
print STDERR ("Found $nqm ? values to be predicted.\n");
print STDERR ("Found $nreg regular non-empty values.\n");
my @languages = keys(%{$devlh});
my $nl = scalar(@languages);
my $sumqm = 0;
my $sumreg = 0;
my $minreg;
my $minreg_qm;
foreach my $l (@languages)
{
    my @features = keys(%{$devlh->{$l}});
    my $nqm = scalar(grep {$devlh->{$l}{$_} eq '?'} (@features));
    my $nreg = scalar(grep {$devlh->{$l}{$_} !~ m/^nan|\?$/} (@features));
    $sumqm += $nqm;
    $sumreg += $nreg;
    if(!defined($minreg) || $nreg<$minreg)
    {
        $minreg = $nreg;
        $minreg_qm = $nqm;
    }
}
my $avgqm = $sumqm/$nl;
my $avgreg = $sumreg/$nl;
print STDERR ("On average, a language has $avgreg non-empty features and $avgqm features to predict.\n");
print STDERR ("Minimum knowledge is $minreg non-empty features; in that case, $minreg_qm features are to be predicted.\n");
print STDERR ("Note that the non-empty features always include 7 non-typologic features: code, name, latitude, longitude, genus, family, countrycodes.\n");
# Predict the masked features.
###!!! This is the first shot...
print STDERR ("Predicting the masked features...\n");
foreach my $l (@languages)
{
    print STDERR ("Language $devlh->{$l}{wals_code} ($devlh->{$l}{name}):\n");
    my @features = keys(%{$devlh->{$l}});
    my @rfeatures = grep {$devlh->{$l}{$_} !~ m/^nan|\?$/} (@features);
    my @qfeatures = grep {$devlh->{$l}{$_} eq '?'} (@features);
    foreach my $qf (@qfeatures)
    {
        print STDERR ("  Predicting $qf:\n");
        my @model;
        foreach my $rf (@rfeatures)
        {
            if(exists($trainprob->{$rf}{$devlh->{$l}{$rf}}{$qf}))
            {
                my @qvalues = keys(%{$trainprob->{$rf}{$devlh->{$l}{$rf}}{$qf}});
                foreach my $qv (@qvalues)
                {
                    push(@model,
                    {
                        'p' => $trainprob->{$rf}{$devlh->{$l}{$rf}}{$qf}{$qv},
                        'c' => $traincooc->{$rf}{$devlh->{$l}{$rf}}{$qf}{$qv},
                        'v' => $qv,
                        'rf' => $rf,
                        'rv' => $devlh->{$l}{$rf}
                    });
                    #print STDERR ("    Cooccurrence with $rf == $devlh->{$l}{$rf} => $qv (p=$trainprob->{$rf}{$devlh->{$l}{$rf}}{$qf}{$qv}).\n");
                }
            }
            else
            {
                #print STDERR ("    No cooccurrence with $rf == $devlh->{$l}{$rf}.\n");
            }
        }
        print STDERR ("    Found ", scalar(@model), " conditional probabilities.\n");
        if(scalar(@model)>0)
        {
            # We want a high probability but we also want it to be based on a sufficiently large count.
            @model = sort {$b->{p}*log($b->{c}) <=> $a->{p}*log($a->{c})} (@model);
            my $prediction = $model[0]{v};
            print STDERR ("    p=$model[0]{p} (count $model[0]{c}) => winner: $prediction (source: $model[0]{rf} == $model[0]{rv})\n");
        }
    }
    exit; ###!!!
}



#------------------------------------------------------------------------------
# Converts the input table to two hashes (returns two hash references).
# The first hash is indexed by features and values; contains number of occur-
# rences. The second hash is indexed by languages and features; contains
# feature values.
#------------------------------------------------------------------------------
sub hash_features
{
    my $headers = shift; # array ref
    my $data = shift; # array ref
    my $qm_is_nan = shift; # convert question marks to 'nan'?
    my %h;
    my %lh; # hash indexed by language code
    foreach my $language (@{$data})
    {
        my $lcode = $language->[1];
        # Remember observed features and values.
        for(my $i = 1; $i <= $#{$headers}; $i++)
        {
            my $feature = $headers->[$i];
            # Make sure that a feature missing from the database is always indicated as 'nan' (normally 'nan' appears already in the input).
            $language->[$i] = 'nan' if(!defined($language->[$i]) || $language->[$i] eq '' || $language->[$i] eq 'nan');
            # Our convention: a question mark masks a feature value that is available in WALS but we want our model to predict it.
            # If desired, we can convert question marks to 'nan' here.
            $language->[$i] = 'nan' if($language->[$i] eq '?' && $qm_is_nan);
            $h{$feature}{$language->[$i]}++;
            $lh{$lcode}{$feature} = $language->[$i];
        }
    }
    return (\%h, \%lh);
}



#------------------------------------------------------------------------------
# Computes pairwise cooccurrence of two feature values in a language.
# Computes conditional probability P(g=gv|f=fv).
# Returns two hash references indexed {f}{fv}{g}{gv}: $cooc and $prob.
#------------------------------------------------------------------------------
sub compute_pairwise_cooccurrence
{
    my $headers = shift; # array ref
    my $lh = shift; # hash ref: features indexed by language
    my @languages = keys(%{$lh});
    my %cooc;
    my %prob;
    foreach my $l (@languages)
    {
        foreach my $f (@{$headers})
        {
            next if(!exists($lh->{$l}{$f}));
            my $fv = $lh->{$l}{$f};
            next if($fv eq 'nan' || $fv eq '?');
            foreach my $g (@{$headers})
            {
                next if($g eq $f);
                next if(!exists($lh->{$l}{$g}));
                my $gv = $lh->{$l}{$g};
                next if($gv eq 'nan' || $gv eq '?');
                $cooc{$f}{$fv}{$g}{$gv}++;
            }
        }
    }
    # Now look at the cooccurrences disregarding individual languages and compute conditional probabilities.
    foreach my $f (@{$headers})
    {
        my @fvalues = keys(%{$cooc{$f}});
        foreach my $fv (@fvalues)
        {
            foreach my $g (keys(%{$cooc{$f}{$fv}}))
            {
                my @gvalues = keys(%{$cooc{$f}{$fv}{$g}});
                # What is the total number of cases when $f=$fv cooccurred with a nonempty value of $g?
                my $nffvg = 0;
                foreach my $gv (@gvalues)
                {
                    $nffvg += $cooc{$f}{$fv}{$g}{$gv};
                }
                foreach my $gv (@gvalues)
                {
                    # Conditional probability of $g=$gv given $f=$fv.
                    $prob{$f}{$fv}{$g}{$gv} = $cooc{$f}{$fv}{$g}{$gv} / $nffvg;
                }
            }
        }
    }
    return (\%cooc, \%prob);
}



#==============================================================================
# Input and output functions.
#==============================================================================



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
    if($myfiles)
    {
        @ARGV = @oldargv;
    }
    return (\@headers, \@data);
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
