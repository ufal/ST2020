#!/usr/bin/env perl
# Reads the training data train_y.csv and computes conditional probabilities of language features.
# Reads the blind development data dev_x.csv, replaces question marks by predicted features and writes the resulting file.
# Copyright © 2020 Dan Zeman <zeman@ufal.mff.cuni.cz>
# License: GNU GPL

# Usage:
#   cd $REPO_ROOT
#   perl scripts/train_and_predict.pl > data/devdz.csv
#   python3 scripts/evaluate_from_csv.py --input_file data/dev_x.csv --output_file data/devdz.csv --golden_file data/dev_y.csv

use utf8;
use open ':utf8';
binmode(STDIN, ':utf8');
binmode(STDOUT, ':utf8');
binmode(STDERR, ':utf8');
use Getopt::Long;

my $debug = 0; # print all predictions with explanation to STDERR
GetOptions
(
    'debug' => \$debug
);

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
# Read the gold standard development data. It will help us with debugging and error analysis.
print STDERR ("Reading the development gold standard data...\n");
my ($devgheaders, $devgdata) = read_csv("$data_folder/dev_y.csv");
print STDERR ("Found ", scalar(@{$devheaders}), " headers.\n");
print STDERR ("Found ", scalar(@{$devdata}), " language lines.\n");
my $ndevlangs = scalar(@{$devdata});
my $ndevfeats = scalar(@{$devheaders})-1; # first column is ord number; except for that, counting everything including the language code and name
my $ndevlangfeats = $ndevlangs*$ndevfeats;
print STDERR ("$ndevlangs languages × $ndevfeats features would be $ndevlangfeats.\n");
my ($devh, $devlh) = hash_features($devheaders, $devdata, 0);
my ($devgh, $devglh) = hash_features($devgheaders, $devgdata, 0);
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
print STDERR ("Note that the non-empty features always include 8 non-typologic features: ord, code, name, latitude, longitude, genus, family, countrycodes.\n");
# Predict the masked features.
###!!! This is the first shot...
print STDERR ("Predicting the masked features...\n");
foreach my $l (@languages)
{
    predict_masked_features($devlh->{$l}, $trainprob, $traincooc, $devglh->{$l});
}
print STDERR ("Writing the completed file...\n");
write_csv($devheaders, $devlh);



#------------------------------------------------------------------------------
# Takes the hash of features of a language, some of the features are masked
# (their value is '?'). Predicts the values of the masked features based on the
# values of the unmasked features. Writes the predicted values directly to the
# hash, i.e., replaces the question marks.
#------------------------------------------------------------------------------
sub predict_masked_features
{
    my $lhl = shift; # hash ref: feature-value hash of one language
    my $prob = shift; # hash ref: conditional probabilities of features given other features
    my $cooc = shift; # hash ref: cooccurrence counts of feature-value pairs
    my $goldlhl = shift; # hash ref: gold standard version of $lhl, used for debugging and analysis
    print STDERR ("Language $lhl->{wals_code} ($lhl->{name}):\n") if($debug);
    my @features = keys(%{$lhl});
    my @rfeatures = grep {$lhl->{$_} !~ m/^nan|\?$/} (@features);
    my @qfeatures = grep {$lhl->{$_} eq '?'} (@features);
    my $nrf = scalar(@rfeatures);
    my $nqf = scalar(@qfeatures);
    print STDERR ("  $nrf features known, $nqf features to be predicted.\n") if($debug);
    foreach my $qf (@qfeatures)
    {
        print STDERR ("  Predicting $qf:\n") if($debug);
        my @model;
        foreach my $rf (@rfeatures)
        {
            if(exists($prob->{$rf}{$lhl->{$rf}}{$qf}))
            {
                my @qvalues = keys(%{$prob->{$rf}{$lhl->{$rf}}{$qf}});
                foreach my $qv (@qvalues)
                {
                    push(@model,
                    {
                        'p' => $prob->{$rf}{$lhl->{$rf}}{$qf}{$qv},
                        'c' => $cooc->{$rf}{$lhl->{$rf}}{$qf}{$qv},
                        'v' => $qv,
                        'rf' => $rf,
                        'rv' => $lhl->{$rf}
                    });
                }
            }
            else
            {
                #print STDERR ("    No cooccurrence with $rf == $lhl->{$rf}.\n");
            }
        }
        print STDERR ("    Found ", scalar(@model), " conditional probabilities.\n") if($debug);
        if(scalar(@model)>0)
        {
            if($debug)
            {
                # Sort source features: the one with strongest possible prediction first.
                my %rfeatures;
                foreach my $cooc (@model)
                {
                    if(!defined($rfeatures{$cooc->{rf}}) || $cooc->{p} > $rfeatures{$cooc->{rf}})
                    {
                        $rfeatures{$cooc->{rf}} = $cooc->{p};
                    }
                }
                my @rfeatures = sort {$rfeatures{$b} <=> $rfeatures{$a}} (keys(%rfeatures));
                @model = sort {$b->{p}*log($b->{c}) <=> $a->{p}*log($a->{c})} (@model);
                foreach my $rfeature (@rfeatures)
                {
                    my $rvalue = $lhl->{$rfeature};
                    # Show all cooccurrences with this rfeature, including the other possible target values, with probabilities.
                    foreach my $cooc (@model)
                    {
                        if($cooc->{rf} eq $rfeature)
                        {
                            print STDERR ("    Cooccurrence with $rfeature == $rvalue => $cooc->{v} (p=$cooc->{p}, c=$cooc->{c}).\n");
                        }
                    }
                }
            }
            # Save the winning prediction in the language-feature hash.
            $lhl->{$qf} = model_take_strongest(@model); # accuracy(dev) = 64.47%
            #$lhl->{$qf} = model_weighted_vote(@model); # accuracy(dev) = 60.28%
            if(defined($goldlhl))
            {
                if($lhl->{$qf} ne $goldlhl->{$qf})
                {
                    print STDERR ("Language $lhl->{name} wrong prediction $qf == $lhl->{$qf}\n");
                    print STDERR ("  should be $goldlhl->{$qf}\n");
                }
            }
        }
    }
}



#------------------------------------------------------------------------------
# Strongest signal: high probability and enough instances the probability is
# based on. We take the prediction suggested by the strongest signal and ignore
# the other suggestions.
#------------------------------------------------------------------------------
sub model_take_strongest
{
    my @model = @_;
    # We want a high probability but we also want it to be based on a sufficiently large count.
    @model = sort {$b->{p}*log($b->{c}) <=> $a->{p}*log($a->{c})} (@model);
    my $prediction = $model[0]{v};
    print STDERR ("    p=$model[0]{p} (count $model[0]{c}) => winner: $prediction (source: $model[0]{rf} == $model[0]{rv})\n") if($debug);
    return $prediction;
}



#------------------------------------------------------------------------------
# We let the individual signals based on available features vote. The votes are
# weighted by the strength of the signal (probability × log count).
#------------------------------------------------------------------------------
sub model_weighted_vote
{
    my @model = @_;
    my %votes;
    foreach my $signal (@model)
    {
        my $weight = $signal->{p}*log($signal->{c});
        $votes{$signal->{v}} += $weight;
    }
    my @options = sort {$votes{$b} <=> $votes{$a}} (keys(%votes));
    my $prediction = $options[0];
    print STDERR ("    $votes{$options[0]}\t$options[0]\n") if($debug);
    return $prediction;
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
        for(my $i = 0; $i <= $#{$headers}; $i++)
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
# Takes the column headers (needed because of their order) and the current
# language-feature hash with the newly predicted values and prints them as
# a CSV file to STDOUT.
#------------------------------------------------------------------------------
sub write_csv
{
    my $headers = shift; # array ref
    my $lh = shift; # hash ref
    my @headers = map {escape_commas($_)} (@{$headers});
    print(join(',', @headers), "\n");
    my @languages = sort {$lh->{$a}{''} <=> $lh->{$b}{''}} (keys(%{$lh}));
    foreach my $l (@languages)
    {
        my @values = map {escape_commas($lh->{$l}{$_})} (@{$headers});
        print(join(',', @values), "\n");
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
