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
my $print_hi = 0; # print entropy of each feature and mutual information of each pair of features
GetOptions
(
    'debug'    => \$debug,
    'print_hi' => \$print_hi
);

#==============================================================================
# The main data structure for a seat of language descriptions read from CSV:
# %data:
#   Filled by read_csv(): -----------------------------------------------------
#     {features} .. names of features = headers of columns in the table
#     {table} ..... the table: array of arrays
#     {nf} ........ number of features (headers)
#     {nl} ........ number of languages
#   Filled by hash_features()
#     {lcodes} .... list of wals codes of known languages (index to lh)
#     {lh} ........ data from {table} indexed by {language}{feature}
#     {lhclean} ... like {lh} but contains only non-empty values (that are not
#                   '', 'nan' or '?')
#     {fcount} .... hash {f} => count of languages where f is not empty
#     {fvcount} ... hash indexed by {feature}{value} => count of that value
#     {fvprob} .... hash {f}{fv} => probability that f=fv
#     {fentropy} .. hash {f} => entropy of distribution of non-empty values of f
#   Filled by compute_pairwise_cooccurrence()
#     {fgcount} ... hash {f}{g} => count of languages where both f and g are not empty
#     {fgvcount} .. hash {f}{g}{gv} => count of g=gv in languages where f is not empty
#     {fgvprob} ... hash {f}{g}{gv} => probability of g=gv given that f is not empty
#     {fgventropy}  hash {f}{g} => entropy of g given that f is not empty
#     {cooc} ...... hash {f}{fv}{g}{gv} => count of languages where f=fv and g=gv
#     {cprob} ..... hash {f}{fv}{g}{gv} => conditional probability(g=gv|f=fv)
#     {jprob} ..... hash {f}{fv}{g}{gv} => joint probability(f=fv, g=gv)
#     {centropy} .. hash {f}{g} => conditional entropy(g|f)
#     {information} ... hash {f}{g} => mutual information between f and g
#==============================================================================

my $data_folder = 'data';
print STDERR ("Reading the training data...\n");
my %traindata = read_csv("$data_folder/train_y.csv");
print STDERR ("Found $traindata{nf} headers.\n");
print STDERR ("Found $traindata{nl} language lines.\n");
print STDERR ("Hashing the features and their cooccurrences...\n");
# Hash the observed features and values.
hash_features(\%traindata, 0);
compute_pairwise_cooccurrence(\%traindata);
if($print_hi)
{
    print_hi(\%traindata);
}
print STDERR ("Reading the development data...\n");
my %devdata = read_csv("$data_folder/dev_x.csv");
# Read the gold standard development data. It will help us with debugging and error analysis.
print STDERR ("Reading the development gold standard data...\n");
my %devgdata = read_csv("$data_folder/dev_y.csv");
print STDERR ("Found $devdata{nf} headers.\n");
print STDERR ("Found $devdata{nl} language lines.\n");
my $ndevlangs = $devdata{nl};
my $ndevfeats = $devdata{nf}-1; # first column is ord number; except for that, counting everything including the language code and name
my $ndevlangfeats = $ndevlangs*$ndevfeats;
print STDERR ("$ndevlangs languages × $ndevfeats features would be $ndevlangfeats.\n");
hash_features(\%devdata, 0);
hash_features(\%devgdata, 0);
print_qm_analysis(\%devdata);
# Predict the masked features.
print STDERR ("Predicting the masked features...\n");
predict_masked_features(\%traindata, \%devdata, \%devgdata);
print STDERR ("Writing the completed file...\n");
write_csv(\%devdata);



#------------------------------------------------------------------------------
# Takes the hash of features of a language, some of the features are masked
# (their value is '?'). Predicts the values of the masked features based on the
# values of the unmasked features. Writes the predicted values directly to the
# hash, i.e., replaces the question marks.
#------------------------------------------------------------------------------
sub predict_masked_features
{
    my $traindata = shift; # hash ref
    my $blinddata = shift; # hash ref
    my $golddata = shift; # hash ref
    my $n_predicted = 0;
    my $n_predicted_correctly = 0;
    # Always process the languages in the same order so that diagnostic outputs can be compared.
    my @lcodes = sort(@{$blinddata->{lcodes}});
    foreach my $language (@lcodes)
    {
        my $lhl = $blinddata->{lh}{$language}; # hash ref: feature-value hash of one language
        my $goldlhl = $golddata->{lh}{$language}; # hash ref: gold standard version of $lhl, used for debugging and analysis
        if($debug)
        {
            print STDERR ("----------------------------------------------------------------------\n");
            print STDERR ("Language $lhl->{wals_code} ($lhl->{name}, $lhl->{family}/$lhl->{genus}, $lhl->{countrycodes}):\n");
        }
        my @features = keys(%{$lhl});
        # Always process the features in the same order. Not only because of diagnostic outputs
        # but also to get the same results in cases where two features both seem to be of same
        # value when predicting a particular third feature.
        my @rfeatures = sort(grep {$lhl->{$_} !~ m/^nan|\?$/} (@features));
        my @qfeatures = sort(grep {$lhl->{$_} eq '?'} (@features));
        my $nrf = scalar(@rfeatures);
        my $nqf = scalar(@qfeatures);
        print STDERR ("  $nrf features known, $nqf features to be predicted.\n") if($debug);
        foreach my $qf (@qfeatures)
        {
            $n_predicted++;
            print STDERR ("  Predicting $qf:\n") if($debug);
            my @model;
            foreach my $rf (@rfeatures)
            {
                if(exists($traindata->{cprob}{$rf}{$lhl->{$rf}}{$qf}))
                {
                    my @qvalues = keys(%{$traindata->{cprob}{$rf}{$lhl->{$rf}}{$qf}});
                    foreach my $qv (@qvalues)
                    {
                        my %record =
                        (
                            'p' => $traindata->{cprob}{$rf}{$lhl->{$rf}}{$qf}{$qv},
                            'c' => $traindata->{cooc}{$rf}{$lhl->{$rf}}{$qf}{$qv},
                            'v' => $qv,
                            'rf' => $rf,
                            'rv' => $lhl->{$rf}
                        );
                        $record{plogc} = $record{p} * log($record{c});
                        $record{plogcinf} = $record{plogc} * $traindata->{information}{$rf}{$qf};
                        push(@model, \%record);
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
                # Save the winning prediction in the language-feature hash.
                #$lhl->{$qf} = model_take_strongest(@model); # accuracy(dev) = 64.47%
                $lhl->{$qf} = model_take_strongest_information(@model); # accuracy(dev) = 69.86%
                #$lhl->{$qf} = model_weighted_vote(@model); # accuracy(dev) = 60.28%
                if(defined($goldlhl))
                {
                    if($lhl->{$qf} eq $goldlhl->{$qf})
                    {
                        $n_predicted_correctly++;
                    }
                    else
                    {
                        print STDERR ("Language $lhl->{name} wrong prediction $qf == $lhl->{$qf}\n");
                        print STDERR ("  should be $goldlhl->{$qf}\n");
                        if($debug)
                        {
                            # Sort source features: the one with strongest possible prediction first.
                            my %rfeatures;
                            foreach my $cooc (@model)
                            {
                                if(!defined($rfeatures{$cooc->{rf}}) || $plogc > $rfeatures{$cooc->{rf}})
                                {
                                    $rfeatures{$cooc->{rf}} = $cooc->{plogcinf};
                                }
                            }
                            my @rfeatures = sort {$rfeatures{$b} <=> $rfeatures{$a}} (keys(%rfeatures));
                            @model = sort {$b->{p}*log($b->{c}) <=> $a->{p}*log($a->{c})} (@model);
                            foreach my $rfeature (@rfeatures)
                            {
                                print STDERR ("    Mutual information with $rfeature == $traindata->{information}{$rfeature}{$qf}\n");
                                my $rvalue = $lhl->{$rfeature};
                                # Show all cooccurrences with this rfeature, including the other possible target values, with probabilities.
                                foreach my $cooc (@model)
                                {
                                    if($cooc->{rf} eq $rfeature)
                                    {
                                        print STDERR ("    Cooccurrence with $rfeature == $rvalue => $cooc->{v} (p=$cooc->{p}, c=$cooc->{c}, plogc=$cooc->{plogc}, plogcinf=$cooc->{plogcinf}).\n");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    print STDERR ("Correctly predicted $n_predicted_correctly features out of $n_predicted total predictions");
    printf STDERR (", accuracy = %.2f%%", $n_predicted_correctly / $n_predicted * 100) unless($n_predicted==0);
    print STDERR ("\n");
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
    @model = sort {$b->{plogc} <=> $a->{plogc}} (@model);
    my $prediction = $model[0]{v};
    print STDERR ("    p=$model[0]{p} (count $model[0]{c}) => winner: $prediction (source: $model[0]{rf} == $model[0]{rv})\n") if($debug);
    return $prediction;
}



#------------------------------------------------------------------------------
# Strongest signal: high probability and enough instances the probability is
# based on. We take the prediction suggested by the strongest signal and ignore
# the other suggestions.
# Here the signal is: cond prob * log count * mutual information
#------------------------------------------------------------------------------
sub model_take_strongest_information
{
    my @model = @_;
    # We want a high probability but we also want it to be based on a sufficiently large count.
    @model = sort {$b->{plogcinf} <=> $a->{plogcinf}} (@model);
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
# Converts the input table to hashes.
#   {lcodes} ...... list of wals codes of known languages (index to lh)
#   {lh} .......... data from {table} indexed by {language}{feature}
#   {lhclean} ..... like {lh} but contains only non-empty values (that are not
#                   '', 'nan' or '?')
#   {fcount} ...... hash {f} => count of languages where f is not empty
#   {fvcount} ..... hash indexed by {feature}{value} => count of that value
#   {fvprob} ...... hash {f}{fv} => probability that f=fv
#   {fentropy} .... hash {f} => entropy of distribution of non-empty values of f
#------------------------------------------------------------------------------
sub hash_features
{
    my $data = shift; # hash ref
    my $qm_is_nan = shift; # convert question marks to 'nan'?
    my @lcodes;
    my %lh; # hash indexed by language code
    my %lhclean; # like %lh but only non-empty values
    my %fcount;
    my %fvcount;
    my %fvprob;
    my %fentropy;
    foreach my $language (@{$data->{table}})
    {
        my $lcode = $language->[1];
        push(@lcodes, $lcode);
        # Remember observed features and values.
        for(my $i = 0; $i <= $#{$data->{features}}; $i++)
        {
            my $feature = $data->{features}[$i];
            # Make sure that a feature missing from the database is always indicated as 'nan' (normally 'nan' appears already in the input).
            $language->[$i] = 'nan' if(!defined($language->[$i]) || $language->[$i] eq '' || $language->[$i] eq 'nan');
            # Our convention: a question mark masks a feature value that is available in WALS but we want our model to predict it.
            # If desired, we can convert question marks to 'nan' here.
            $language->[$i] = 'nan' if($language->[$i] eq '?' && $qm_is_nan);
            $lh{$lcode}{$feature} = $language->[$i];
            unless($language->[$i] eq 'nan' || $language->[$i] eq '?')
            {
                $lhclean{$lcode}{$feature} = $language->[$i];
                $fcount{$feature}++;
                $fvcount{$feature}{$language->[$i]}++;
            }
        }
    }
    # Compute unconditional probability of each feature value and entropy of each feature.
    foreach my $f (@{$data->{features}})
    {
        $fentropy{$f} = 0;
        next if($fcount{$f}==0);
        foreach my $fv (keys(%{$fvcount{$f}}))
        {
            my $p = $fvcount{$f}{$fv} / $fcount{$f};
            if($p < 0 || $p > 1)
            {
                die("Something is wrong: p = $p");
            }
            $fvprob{$f}{$fv} = $p;
            $fentropy{$f} -= $p * log($p);
        }
    }
    $data->{lcodes} = \@lcodes;
    $data->{lh} = \%lh;
    $data->{lhclean} = \%lhclean;
    $data->{fcount} = \%fcount;
    $data->{fvcount} = \%fvcount;
    $data->{fvprob} = \%fvprob;
    $data->{fentropy} = \%fentropy;
}



#------------------------------------------------------------------------------
# Computes pairwise cooccurrence of two feature values in a language.
# Computes conditional probability P(g=gv|f=fv).
# Filled by compute_pairwise_cooccurrence()
#   {fgcount} ... hash {f}{g} => count of languages where both f and g are not empty
#   {fgvcount} .. hash {f}{g}{gv} => count of g=gv in languages where f is not empty
#   {fgvprob} ... hash {f}{g}{gv} => probability of g=gv given that f is not empty
#   {fgventropy}  hash {f}{g} => entropy of g given that f is not empty
#   {cooc} ...... hash {f}{fv}{g}{gv} => count of languages where f=fv and g=gv
#   {cprob} ..... hash {f}{fv}{g}{gv} => conditional probability(g=gv|f=fv)
#   {jprob} ..... hash {f}{fv}{g}{gv} => joint probability(f=fv, g=gv)
#   {centropy} .. hash {f}{g} => conditional entropy(g|f)
#   {information} ... hash {f}{g} => mutual information between f and g
#------------------------------------------------------------------------------
sub compute_pairwise_cooccurrence
{
    my $data = shift; # hash ref
    my %fgcount;
    my %fgvcount;
    my %fgvprob;
    my %fgventropy;
    my %cooc;
    my %prob;
    my %jprob;
    my %centropy;
    my %information;
    foreach my $l (@{$data->{lcodes}})
    {
        foreach my $f (@{$data->{features}})
        {
            next if(!exists($data->{lhclean}{$l}{$f}));
            my $fv = $data->{lhclean}{$l}{$f};
            foreach my $g (@{$data->{features}})
            {
                next if($g eq $f);
                next if(!exists($data->{lhclean}{$l}{$g}));
                my $gv = $data->{lhclean}{$l}{$g};
                $fgcount{$f}{$g}++;
                $fgvcount{$f}{$g}{$gv}++;
                $cooc{$f}{$fv}{$g}{$gv}++;
            }
        }
    }
    # Now look at the cooccurrences disregarding individual languages and compute
    # conditional probabilities, joint probabilities, and conditional entropy.
    foreach my $f (@{$data->{features}})
    {
        my @fvalues = keys(%{$cooc{$f}});
        foreach my $fv (@fvalues)
        {
            foreach my $g (keys(%{$cooc{$f}{$fv}}))
            {
                # Probability of f=fv given that g is not empty. We will need
                # the entropy of this distribution to compute mutual information
                # of f and g.
                my $pfv = $fgvcount{$g}{$f}{$fv} / $fgcount{$f}{$g};
                $fgvprob{$g}{$f}{$fv} = $pfv;
                if($pfv > 0)
                {
                    $fgventropy{$g}{$f} -= $pfv * log($pfv);
                }
                my @gvalues = keys(%{$cooc{$f}{$fv}{$g}});
                foreach my $gv (@gvalues)
                {
                    # Conditional probability of $g=$gv given $f=$fv.
                    $prob{$f}{$fv}{$g}{$gv} = $cooc{$f}{$fv}{$g}{$gv} / $fgvcount{$g}{$f}{$fv};
                    # Joint probability of $f=$fv and $g=$gv.
                    my $pfvgv = $cooc{$f}{$fv}{$g}{$gv} / $fgcount{$f}{$g};
                    # Conditional entropy of $g given $f.
                    if($pfvgv > 0 && $pfv > 0)
                    {
                        $centropy{$f}{$g} -= $pfvgv * log($pfvgv/$pfv);
                    }
                    $jprob{$f}{$fv}{$g}{$gv} = $pfvgv;
                }
            }
        }
    }
    # Once we have conditional entropies we can compute mutual information.
    foreach my $f (keys(%centropy))
    {
        foreach my $g (keys(%{$centropy{$f}}))
        {
            # $fgventropy{$f}{$g} is H(g|f is not empty)
            # $centropy{$f}{$g} is H(g|f), i.e., entropy of g given f.
            # Mutual information of $f and $g in the domain where both are not empty
            # (we do not want to recognize the empty value as something that can be predicted):
            $information{$f}{$g} = $fgventropy{$f}{$g} - $centropy{$f}{$g};
            ###!!! Sanity check.
            # Due to imprecise computation with extremely small numbers, sometimes
            # we get very slightly below zero. Let's tolerate it and let's treat
            # the same interval above zero symmetrically.
            if($information{$f}{$g} > -1e-15 && $information{$f}{$g} < 1e-15)
            {
                $information{$f}{$g} = 0;
            }
            if($information{$f}{$g} < 0)
            {
                print STDERR ("Something is wrong. Mutual information must not be negative but it is I = $information{$f}{$g}\n");
                print STDERR ("\tf = $f\n");
                print STDERR ("\tg = $g\n");
                print STDERR ("\tH(g) = $fgventropy{$f}{$g} (only in languages where f is not empty)\n");
                print STDERR ("\tH(g|f) = $centropy{$f}{$g}\n");
                print STDERR ("\t\t$data->{fcount}{$f} = number of nonempty occurrences of f\n");
                print STDERR ("\t\t$data->{fcount}{$g} = number of nonempty occurrences of g\n");
                print STDERR ("\t\t$fgcount{$f}{$g} = number of nonempty cooccurrences of f and g\n");
                die;
            }
        }
    }
    $data->{fgcount} = \%fgcount;
    $data->{fgvcount} = \%fgvcount;
    $data->{fgvprob} = \%fgvprob;
    $data->{fgventropy} = \%fgventropy;
    $data->{cooc} = \%cooc;
    $data->{cprob} = \%prob;
    $data->{jprob} = \%jprob;
    $data->{centropy} = \%centropy;
    $data->{information} = \%information;
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
    my $data = shift; # hash ref
    my $lh = $data->{lh}; # hash ref
    my @headers = map {escape_commas($_)} (@{$data->{features}});
    print(join(',', @headers), "\n");
    my @languages = sort {$lh->{$a}{index} <=> $lh->{$b}{index}} (keys(%{$lh}));
    foreach my $l (@languages)
    {
        my @values = map {escape_commas($lh->{$l}{$_})} (@{$data->{features}});
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
    if(scalar(@headers) > 0 && $headers[0] eq '')
    {
        $headers[0] = 'index';
    }
    my %data =
    (
        'features' => \@headers,
        'table'    => \@data,
        'nf'       => scalar(@headers),
        'nl'       => scalar(@data)
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



#------------------------------------------------------------------------------
# Prints entropy of each feature and mutual information of each pair of
# features to STDERR.
#------------------------------------------------------------------------------
sub print_hi
{
    my $data = shift;
    print STDERR ("Entropy of each feature:\n");
    my @features_by_entropy = sort {$data->{fentropy}{$a} <=> $data->{fentropy}{$b}} (@{$data->{features}});
    foreach my $feature (@features_by_entropy)
    {
        print STDERR ("  $data->{fentropy}{$feature} = H($feature)\n");
    }
    print STDERR ("Mutual information of each pair of features:\n");
    my @feature_pairs;
    foreach my $f (keys(%{$data->{centropy}}))
    {
        next if($f =~ m/^(index|wals_code|name)$/);
        foreach my $g (keys(%{$data->{centropy}{$f}}))
        {
            next if($g =~ m/^(index|wals_code|name)$/ || $g le $f);
            push(@feature_pairs, [$f, $g]);
        }
    }
    my @feature_pairs_by_information = sort {$data->{information}{$b->[0]}{$b->[1]} <=> $data->{information}{$a->[0]}{$a->[1]}} (@feature_pairs);
    foreach my $fg (@feature_pairs_by_information)
    {
        print STDERR ("  $data->{information}{$fg->[0]}{$fg->[1]} = I( $fg->[0] , $fg->[1] )\n");
    }
}



#------------------------------------------------------------------------------
# Analyzes the question marks in data: how much do we know and how much do we
# have to predict? Prints the findings to STDERR.
#------------------------------------------------------------------------------
sub print_qm_analysis
{
    my $data = shift;
    my @features = @{$data->{features}};
    my $nnan = 0;
    my $nqm = 0;
    my $nreg = 0;
    foreach my $f (@features)
    {
        my @values = keys(%{$data->{fvcount}{$f}});
        foreach my $v (@values)
        {
            if($v eq 'nan')
            {
                $nnan += $data->{fvcount}{$f}{$v};
            }
            elsif($v eq '?')
            {
                $nqm += $data->{fvcount}{$f}{$v};
            }
            else
            {
                $nreg += $data->{fvcount}{$f}{$v};
            }
        }
    }
    print STDERR ("Found $nnan nan values.\n");
    print STDERR ("Found $nqm ? values to be predicted.\n");
    print STDERR ("Found $nreg regular non-empty values.\n");
    my @languages = @{$data->{lcodes}};
    my $nl = $data->{nl};
    my $sumqm = 0;
    my $sumreg = 0;
    my $minreg;
    my $minreg_qm;
    my $minreg_langname;
    foreach my $l (@languages)
    {
        my @features = keys(%{$data->{lh}{$l}});
        my $nqm = scalar(grep {$data->{lh}{$l}{$_} eq '?'} (@features));
        my $nreg = scalar(grep {$data->{lh}{$l}{$_} !~ m/^nan|\?$/} (@features));
        $sumqm += $nqm;
        $sumreg += $nreg;
        if(!defined($minreg) || $nreg<$minreg)
        {
            $minreg = $nreg;
            $minreg_qm = $nqm;
            $minreg_langname = $data->{lh}{name};
        }
    }
    my $avgqm = $sumqm/$nl;
    my $avgreg = $sumreg/$nl;
    print STDERR ("On average, a language has $avgreg non-empty features and $avgqm features to predict.\n");
    print STDERR ("Minimum knowledge is $minreg non-empty features (language $minreg_langname); in that case, $minreg_qm features are to be predicted.\n");
    print STDERR ("Note that the non-empty features always include 8 non-typologic features: ord, code, name, latitude, longitude, genus, family, countrycodes.\n");
}
