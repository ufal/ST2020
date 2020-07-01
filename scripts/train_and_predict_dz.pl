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
# We need to tell Perl where to find my IO module.
# If this does not work, you can put the script together with Sigtypio.pm
# in a folder of you choice, say, /home/joe/scripts, and then
# invoke Perl explicitly telling it where the modules are:
# perl -I/home/joe/scripts /home/joe/scripts/train_and_predict_dz.pl ...
BEGIN
{
    use Cwd;
    my $path = $0;
    $path =~ s/\\/\//g;
    #print STDERR ("path=$path\n");
    my $currentpath = getcwd();
    $currentpath =~ s/\r?\n$//;
    $libpath = $currentpath;
    if($path =~ m:/:)
    {
        $path =~ s:/[^/]*$:/:;
        chdir($path);
        $libpath = getcwd();
        chdir($currentpath);
    }
    $libpath =~ s/\r?\n$//;
    #print STDERR ("libpath=$libpath\n");
}
use lib $libpath;
use Sigtypio;

my %config;
# Possible debug levels:
#   0 ... print only progress information and accuracy to STDERR
#   1 ... print all predictions and whether they are correct
#   2 ... print also other options so we can see why a wrong prediction was made
$config{debug} = 0;
$config{print_hi} = 0; # print entropy of each feature and mutual information of each pair of features
# Possible train-on-dev values:
#   no ... do not use any information from the development data when training
#   blind (default) ... use the information that was left visible in the blinded development data
#   gold ... use the full development set as additional training data
$config{train_on_dev} = 'blind';
# Possible scores:
#   c ... how many times we observed that target feature = x and source feature = y in the same language
#   p ... conditional probability that target feature = x given that source feature = y
#   plogc ... p × log(c) ... we want high probability but we also want it to be based on a sufficiently large count
#   information ... mutual information between source and target features
#   plogcinf ... plogc × mutual information between source and target features
$config{score} = 'plogcinf'; # what number should be used to score predictions
# Possible models:
#   strongest ... take the prediction with the highest score, ignore the others
#   vote ... each prediction is counted as a vote weighted by its score; the prediction with most votes wins
#   infovote ... take three source features with highest mutual information with the target feature; let their best predictions vote, ignore the rest
$config{model} = 'strongest'; # what model should be used to convert the scores into one winning prediction
# The feature countrycodes is unreliable. It says US (United States) for a heterogenous
# group of languages, many of them from other continents (e.g., Africa).
# --countrycodes nous ... whenever countrycodes=US, replace it by nan
# We could also remove countrycodes completely but we have not tried this yet.
$config{countrycodes} = '';
# The features latitude and longitude are hardly useful for inference directly.
# However, grouping languages in geographical zones may help.
# --latlon zones ... both latitude and longitude grouped into wide zones around the globe
$config{latlon} = '';
# --latlon2d ... combine latitude and longitude into a 2D area (this only works with --latlon zones)
$config{latlon2d} = 0;
# Similarly to the combination of latitude and longitude above, other features
# may also be stronger in combination than in isolation. We can create a new
# feature for each pair of non-empty features. This will be done after latlon2d,
# so latlon can enter pairs as a single feature.
# --feats2d ... switch on pairs of features
$config{feats2d} = 0;
# Base name of the output files.
$config{name} = 'dz';
GetOptions
(
    'debug=i'  => \$config{debug},
    'print_hi' => \$config{print_hi},
    'train_on_dev=s' => \$config{train_on_dev},
    'score=s'  => \$config{score},
    'model=s'  => \$config{model},
    'countrycodes=s' => \$config{countrycodes},
    'latlon=s' => \$config{latlon},
    'latlon2d' => \$config{latlon2d},
    'feats2d'  => \$config{feats2d},
    'name=s'   => \$config{name}
);

#==============================================================================
# The main data structure for a set of language descriptions read from CSV:
# %data:
#   Filled by read_csv(): -----------------------------------------------------
#     {features} .. names of features = headers of columns in the table
#     {table} ..... the table: array of arrays
#     {nf} ........ number of features (headers)
#     {nl} ........ number of languages
#   Filled by hash_features()
#     {lcodes} .... list of wals codes of known languages (index to lh)
#     {lh} ........ data from {table} indexed by {language}{feature}
#     {restore} ... like {lh} but only original values where we modified them
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
#   Filled by predict_masked_features()
#     {scores} .... hash {language}{feature} => score of the predicted value in lh
#==============================================================================

my $data_folder = 'data';
my $output_folder = 'outputs';
my $wals_folder = 'data/wals-2020/cldf';
my %wals;
if(-d $wals_folder)
{
    print STDERR ("Reading WALS...\n");
    %wals = Sigtypio::read_wals($wals_folder);
    $wals{loaded} = 1;
}
print STDERR ("Reading the training data...\n");
my %traindata = Sigtypio::read_csv("$data_folder/train_y.csv");
Sigtypio::convert_table_to_lh(\%traindata, 0);
print STDERR ("Found $traindata{nf} headers.\n");
print STDERR ("Found $traindata{nl} language lines.\n");
print STDERR ("Reading the development data...\n");
my %devdata = Sigtypio::read_csv("$data_folder/dev_x.csv");
Sigtypio::convert_table_to_lh(\%devdata, 0);
# Read the gold standard development data. It will help us with debugging and error analysis.
print STDERR ("Reading the development gold standard data...\n");
my %devgdata = Sigtypio::read_csv("$data_folder/dev_y.csv");
Sigtypio::convert_table_to_lh(\%devgdata, 0);
print STDERR ("Found $devdata{nf} headers.\n");
print STDERR ("Found $devdata{nl} language lines.\n");
print STDERR ("Reading the blind test data...\n");
my %testdata = Sigtypio::read_csv("$data_folder/test_x.csv");
Sigtypio::convert_table_to_lh(\%testdata, 0);
print STDERR ("Found $testdata{nf} headers.\n");
print STDERR ("Found $testdata{nl} language lines.\n");
# First compare, then merge. Otherwise the comparing function will complain that a language occurs in both sets.
print STDERR ("Comparing training and development data...\n");
compare_data_sets(\%traindata, \%devdata, \%wals);
print STDERR ("Comparing training and test data...\n");
compare_data_sets(\%traindata, \%testdata, \%wals);
if($config{train_on_dev} eq 'blind')
{
    merge_data(\%traindata, \%devdata);
}
elsif($config{train_on_dev} eq 'gold')
{
    merge_data(\%traindata, \%devgdata);
}
merge_data(\%traindata, \%testdata);
# Everything is read. Now organize the data better.
print STDERR ("Hashing the features and their cooccurrences...\n");
# Hash the observed features and values.
hash_features(\%traindata);
compute_pairwise_cooccurrence(\%traindata);
if($config{print_hi})
{
    print_hi(\%traindata);
}
my $ndevlangs = $devdata{nl};
my $ndevfeats = $devdata{nf}-1; # first column is ord number; except for that, counting everything including the language code and name
my $ndevlangfeats = $ndevlangs*$ndevfeats;
print STDERR ("$ndevlangs languages × $ndevfeats features would be $ndevlangfeats.\n");
hash_features(\%devdata);
hash_features(\%devgdata);
print_qm_analysis(\%devdata);
# Predict the masked features.
print STDERR ("Predicting the masked features...\n");
predict_masked_features(\%traindata, \%devdata, \%devgdata);
print STDERR ("Writing the completed file...\n");
Sigtypio::write_csv(\%devdata, "$output_folder/$config{name}-dev.csv");
Sigtypio::write_scores(\%devdata, "$output_folder/$config{name}-dev-scores.csv");
my $ntestlangs = $testdata{nl};
my $ntestfeats = $testdata{nf}-1; # first column is ord number; except for that, counting everything including the language code and name
my $ntestlangfeats = $ntestlangs*$ntestfeats;
print STDERR ("$ntestlangs languages × $ntestfeats features would be $ntestlangfeats.\n");
hash_features(\%testdata);
print_qm_analysis(\%testdata);
# Predict the masked features.
print STDERR ("Predicting the masked features...\n");
predict_masked_features(\%traindata, \%testdata);
print STDERR ("Writing the completed file...\n");
Sigtypio::write_csv(\%testdata, "$output_folder/$config{name}-test.csv");
Sigtypio::write_scores(\%testdata, "$output_folder/$config{name}-test-scores.csv");



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
    my $golddata = shift; # hash ref; may be undefined for evaluation test data
    # Variables used to evaluate the accuracy (total, per language, per feature).
    my $n_predicted = 0;
    my $n_predicted_correctly = 0;
    my %leval; # $leval{$lcode}{correct|total}
    my %feval; # $feval{$feature}{correct|total}
    # We will also save the scores of the answers so that we can later assess their credibility.
    my %scores; # $scores{$lcode}{$feature} = $score
    $blinddata->{scores} = \%scores;
    # Always process the languages in the same order so that diagnostic outputs can be compared.
    my @lcodes = sort(@{$blinddata->{lcodes}});
    foreach my $language (@lcodes)
    {
        my $lhl = $blinddata->{lh}{$language}; # hash ref: feature-value hash of one language
        my $goldlhl;
        if(defined($golddata))
        {
            $goldlhl = $golddata->{lh}{$language}; # hash ref: gold standard version of $lhl, used for debugging and analysis
        }
        if($config{debug})
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
        print STDERR ("  $nrf features known, $nqf features to be predicted.\n") if($config{debug});
        foreach my $qf (@qfeatures)
        {
            $n_predicted++;
            $leval{$language}{total}++;
            $feval{$qf}{total}++;
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
                        $record{information} = $traindata->{information}{$rf}{$qf};
                        # The global parameter $config{score} identifies the number that should be used to score predictions.
                        if(exists($record{$config{score}}))
                        {
                            $record{score} = $record{$config{score}};
                        }
                        else
                        {
                            die("Unknown scoring function '$config{score}'");
                        }
                        push(@model, \%record);
                    }
                }
                else
                {
                    #print STDERR ("    No cooccurrence with $rf == $lhl->{$rf}.\n");
                }
            }
            print STDERR ("    Found ", scalar(@model), " conditional probabilities.\n") if($config{debug} >= 2);
            if(scalar(@model)>0)
            {
                # Save the winning prediction in the language-feature hash.
                # The global parameter $config{model} tells us how to use the scores to obtain the winner.
                if($config{model} eq 'strongest')
                {
                    ($lhl->{$qf}, $blinddata->{scores}{$language}{$qf}) = model_take_strongest(@model);
                }
                elsif($config{model} eq 'threshold')
                {
                    ($lhl->{$qf}, $blinddata->{scores}{$language}{$qf}) = model_strongest_threshold(@model);
                }
                elsif($config{model} eq 'vote')
                {
                    $lhl->{$qf} = model_weighted_vote(@model);
                }
                elsif($config{model} eq 'infovote')
                {
                    $lhl->{$qf} = model_information_vote(@model);
                }
                else
                {
                    die("Unknown model type '$config{model}'");
                }
                if(defined($goldlhl))
                {
                    if($lhl->{$qf} eq $goldlhl->{$qf})
                    {
                        $n_predicted_correctly++;
                        $leval{$language}{correct}++;
                        $feval{$qf}{correct}++;
                        if($config{debug} >= 1)
                        {
                            print STDERR ("Language $lhl->{name} feature $qf:\n");
                            print STDERR ("  correctly predicted == $lhl->{$qf}\n");
                        }
                        if($config{debug} >= 3)
                        {
                            # By grepping the SCORESULT lines, we can assess correlation between scores and correctness of the prediction.
                            print STDERR ("SCORESULT CORRECT $blinddata->{scores}{$language}{$qf}\n");
                        }
                    }
                    else
                    {
                        if($config{debug} >= 1)
                        {
                            print STDERR ("Language $lhl->{name} feature $qf:\n");
                            print STDERR ("  wrongly predicted   == $lhl->{$qf}\n");
                            print STDERR ("  should be           == $goldlhl->{$qf}\n");
                        }
                        if($config{debug} >= 3)
                        {
                            # By grepping the SCORESULT lines, we can assess correlation between scores and correctness of the prediction.
                            print STDERR ("SCORESULT WRONG   $blinddata->{scores}{$language}{$qf}\n");
                        }
                        if($config{debug} >= 2)
                        {
                            if(!is_gold_value_reachable(\@model, $goldlhl->{$qf}))
                            {
                                print STDERR ("The gold-standard value is not reachable given the available features!\n");
                            }
                            if(!is_gold_value_predictable(\@model, $goldlhl->{$qf}))
                            {
                                print STDERR ("The gold-standard value is not predictable (there is always another value with a higher score).\n");
                            }
                            # Sort source features: the one with strongest possible prediction first.
                            my %rfeatures;
                            foreach my $item (@model)
                            {
                                if(!defined($rfeatures{$item->{rf}}) || $item->{score} > $rfeatures{$item->{rf}})
                                {
                                    $rfeatures{$item->{rf}} = $item->{score};
                                }
                            }
                            my @rfeatures = sort {$rfeatures{$b} <=> $rfeatures{$a}} (keys(%rfeatures));
                            @model = sort_model(@model);
                            foreach my $rfeature (@rfeatures)
                            {
                                print STDERR ("    Mutual information with $rfeature == $traindata->{information}{$rfeature}{$qf}\n");
                                my $rvalue = $lhl->{$rfeature};
                                # Show all cooccurrences with this rfeature, including the other possible target values, with probabilities.
                                foreach my $item (@model)
                                {
                                    if($item->{rf} eq $rfeature)
                                    {
                                        print STDERR ("      Cooccurrence with $rfeature == $rvalue => $item->{v} (score=$item->{score}, p=$item->{p}, c=$item->{c}, plogc=$item->{plogc}, plogcinf=$item->{plogcinf}).\n");
                                    }
                                }
                            }
                        }
                    }
                }
                else # no gold standard data available
                {
                    if($config{debug} >= 1)
                    {
                        print STDERR ("Language $lhl->{name} feature $qf:\n");
                        print STDERR ("  predicted == $lhl->{$qf}\n");
                    }
                }
            }
        }
    }
    if(defined($golddata))
    {
        print STDERR ("Correctly predicted $n_predicted_correctly features out of $n_predicted total predictions");
        printf STDERR (", accuracy = %.2f%%", $n_predicted_correctly / $n_predicted * 100) unless($n_predicted==0);
        print STDERR ("\n");
        print STDERR ("Per-language accuracy:\n");
        my @lcodes = keys(%leval);
        foreach my $lcode (@lcodes)
        {
            $leval{$lcode}{accuracy} = $leval{$lcode}{correct} / $leval{$lcode}{total} * 100 unless($leval{$lcode}{total}==0);
        }
        @lcodes = sort
        {
            my $r = $leval{$b}{accuracy} <=> $leval{$a}{accuracy};
            unless($r)
            {
                $r = $a cmp $b;
            }
            $r
        }
        (@lcodes);
        foreach my $lcode (@lcodes)
        {
            printf STDERR ("  $lcode: %3d / %3d = %.2f%% : %s (%s/%s)\n", $leval{$lcode}{correct}, $leval{$lcode}{total}, $leval{$lcode}{accuracy}, $blinddata->{lh}{$lcode}{name}, $blinddata->{lh}{$lcode}{family}, $blinddata->{lh}{$lcode}{genus});
        }
        print STDERR ("Per-feature accuracy:\n");
        my @features = keys(%feval);
        foreach my $feature (@features)
        {
            $feval{$feature}{accuracy} = $feval{$feature}{correct} / $feval{$feature}{total} * 100 unless($feval{$feature}{total}==0);
        }
        @features = sort
        {
            my $r = $feval{$b}{accuracy} <=> $feval{$a}{accuracy};
            unless($r)
            {
                $r = $a cmp $b;
            }
            $r
        }
        (@features);
        foreach my $feature (@features)
        {
            printf STDERR ("  %3d / %3d = %.2f%% : %s\n", $feval{$feature}{correct}, $feval{$feature}{total}, $feval{$feature}{accuracy}, $feature);
        }
    }
    else
    {
        print STDERR ("Predicted $n_predicted feature values. Accuracy is unknown because we cannot access the gold-standard data.\n");
    }
}



#------------------------------------------------------------------------------
# Takes a model and a gold-standard target feature-value pair. Looks for source
# feature-value pairs that have been observed together with the target pair and
# can thus provide signals for the correct prediction.
#------------------------------------------------------------------------------
sub is_gold_value_reachable
{
    my $model = shift; # array reference; the model for the queried feature
    my $goldv = shift; # gold-standard value of the queried feature
    foreach my $item (@{$model})
    {
        if($item->{v} eq $goldv && $item->{c} > 0)
        {
            return 1;
        }
    }
    return 0;
}



#------------------------------------------------------------------------------
# A more demanding option: the gold-standard value must have the best score
# with at least one other feature.
#------------------------------------------------------------------------------
sub is_gold_value_predictable
{
    my $model = shift; # array reference; the model for the queried feature
    my $goldv = shift; # gold-standard value of the queried feature
    # For each real feature, check the score of the gold-standard value.
    my %rfeatures;
    my %goldv_scores_for_rfs;
    foreach my $item (@{$model})
    {
        $rfeatures{$item->{rf}}++;
        if($item->{v} eq $goldv && $item->{c} > 0)
        {
            $goldv_scores_for_rfs{$item->{rf}} = $item->{score};
        }
    }
    # Now for each real feature check whether it scores higher something else.
    # If not, it is a useful predictor.
    my %notgood;
    foreach my $item (@{$model})
    {
        if($item->{v} ne $goldv && exists($goldv_scores_for_rfs{$item->{rf}}) && $item->{score} > $goldv_scores_for_rfs{$item->{rf}})
        {
            $notgood{$item->{rf}}++;
        }
    }
    # Number of available source features:
    my $nrf = scalar(keys(%rfeatures));
    # Number of source features which have been observed with the correct target value:
    my $nrf1 = scalar(keys(%goldv_scores_for_rfs));
    # Number of source features with which the correct target value has been observed but does not have the best score among possible target values:
    my $nng = scalar(keys(%notgood));
    # Number of source features with which the correct target value has the best score among possible target values:
    my $nok = $nrf1 - $nng;
    if($nok > 0)
    {
        print STDERR ("    The correct value '$goldv' has a better score than the wrong values with the following $nok features (out of total $nrf known features):\n");
        foreach my $rf (keys(%goldv_scores_for_rfs))
        {
            if(!exists($notgood{$rf}))
            {
                print STDERR ("      $rf\n");
            }
        }
    }
    return $nok;
}



#------------------------------------------------------------------------------
# Sorts cooccurrence items in model. Tries to make the order stable across runs
# i.e., not dependent on hash key ordering.
#------------------------------------------------------------------------------
sub sort_model
{
    return sort
    {
        # The main sorting is by score, descending order.
        my $result = $b->{score} <=> $a->{score};
        # The rest is just to ensure always the same order in case of ties.
        unless($result)
        {
            # By source feature name.
            $result = $a->{rf} cmp $b->{rf};
            unless($result)
            {
                # By target value.
                $result = $a->{v} cmp $b->{v};
            }
        }
        $result
    }
    (@_);
}



#------------------------------------------------------------------------------
# Strongest signal: Take the prediction with the highest score, ignore others.
#------------------------------------------------------------------------------
sub model_take_strongest
{
    my @model = sort_model(@_);
    my $prediction = $model[0]{v};
    my $score = $model[0]{score};
    return ($prediction, $score);
}



#------------------------------------------------------------------------------
# Conditional vote. Take the strongest signal but if it is not strong enough
# (score < threshold), organize a vote among three strongest signals.
#------------------------------------------------------------------------------
sub model_strongest_threshold
{
    my $threshold = 2.1; # only do something special if score < $threshold
    my $factor = 1.5; # second + third must be together better than $factor * first
    my @model = sort_model(@_);
    my $prediction = $model[0]{v};
    my $score = $model[0]{score};
    if($score < $threshold && scalar(@model) >= 3 && $model[1]{v} ne $model[0]{v} && $model[2]{v} eq $model[1]{v})
    {
        my $score12 = $model[1]{score} + $model[2]{score};
        if($score12 > $score * $factor)
        {
            $prediction = $model[1]{v};
            $score = $score12;
        }
    }
    return ($prediction, $score);
}



#------------------------------------------------------------------------------
# We let the individual signals based on available features vote. The votes are
# weighted by the strength of the signal (score).
#------------------------------------------------------------------------------
sub model_weighted_vote
{
    my @model = @_;
    my %votes;
    foreach my $item (@model)
    {
        $votes{$item->{v}} += $item->{score};
    }
    my @options = sort {$votes{$b} <=> $votes{$a}} (keys(%votes));
    my $prediction = $options[0];
    return $prediction;
}



#------------------------------------------------------------------------------
# Listen to source features with maximal mutual information.
#------------------------------------------------------------------------------
sub model_information_vote
{
    my @model = sort_model(@_);
    # Extract source feature names and sort them by mutual information.
    my %rfeatures; map {$rfeatures{$_->{rf}} = $_->{information}} (@model);
    my @rfeatures = sort {my $r = $rfeatures{$b} <=> $rfeatures{$a}; $r = $a cmp $b unless($r); $r} (keys(%rfeatures));
    # Extract the best prediction (given the current $config{score}) for each source feature.
    my %predictions;
    my %scores;
    foreach my $item (@model) # model is already sorted
    {
        if(!exists($predictions{$item->{rf}}))
        {
            $predictions{$item->{rf}} = $item->{v};
            $scores{$item->{rf}} = $item->{score};
        }
    }
    # Keep the three most informed source features (or a few more if they have the same information value).
    if(scalar(@rfeatures) > 3)
    {
        my $threshold = $rfeatures{$rfeatures[2]};
        @rfeatures = grep {$rfeatures{$_} >= $threshold} (@rfeatures);
    }
    # Let the surviving features vote about the outcome. The vote will be weighted by the score of the prediction.
    my %votes;
    foreach my $rf (@rfeatures)
    {
        $votes{$predictions{$rf}} += $scores{$rf};
    }
    my @options = sort {my $r = $votes{$b} <=> $votes{$a}; $r = $a cmp $b unless($r); $r} (keys(%votes));
    my $prediction = $options[0];
    return $prediction;
}



#------------------------------------------------------------------------------
# Merges two datasets. Looks at {lh} and not at later more sophisticated
# hashes, so it should be called early. Can be used to add non-blinded
# information from the dev/test set to the training data. The first data set
# is the target where additional information from the second data set will be
# added.
#------------------------------------------------------------------------------
sub merge_data
{
    my $d1 = shift; # hash ref
    my $d2 = shift; # hash ref
    my $n = scalar(keys(%{$d1->{lh}}));
    #print STDERR ("The first data set contains $n languages before merging.\n");
    foreach my $lcode (@{$d2->{lcodes}})
    {
        # No language should originally be in both data sets. If it happens,
        # ignore the language, i.e., keep its values from the target set.
        unless(exists($d1->{lh}{$lcode}))
        {
            push(@{$d1->{lcodes}}, $lcode);
            foreach my $feature (@{$d1->{features}})
            {
                my $value = $d2->{lh}{$lcode}{$feature};
                # Make sure that a feature missing from the database is always indicated as 'nan' (normally 'nan' appears already in the input).
                # In this function we also handle all question marks as empty values (we are copying test data but only the non-blind/input part of it).
                $value = 'nan' if(!defined($value) || $value eq '' || $value eq 'nan' || $value eq '?');
                $d1->{lh}{$lcode}{$feature} = $value;
                #print STDERR ("MERGE: language $lcode, $feature = $value\n") unless($value eq 'nan');
            }
        }
    }
    $n = scalar(keys(%{$d1->{lh}}));
    #print STDERR ("The first data set contains $n languages after merging.\n");
}



#------------------------------------------------------------------------------
# Reads the initial hash {lh} and indexes the data by further hashes.
#   {restore} ..... like {lh} but only original values where we modified them
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
    my %lhclean; # like %lh but only non-empty values
    my %fcount;
    my %fvcount;
    my %fvprob;
    my %fentropy;
    # Modify features to improve prediction.
    modify_features($data);
    # Now go on and fill the other hashes that depend solely on one language-feature-value triple.
    foreach my $lcode (@{$data->{lcodes}})
    {
        # Remember observed features and values.
        foreach my $feature (@{$data->{features}})
        {
            my $value = $data->{lh}{$lcode}{$feature};
            unless($value eq 'nan' || $value eq '?')
            {
                $lhclean{$lcode}{$feature} = $value;
                $fcount{$feature}++;
                $fvcount{$feature}{$value}++;
            }
        }
    }
    # Compute unconditional probability of each feature value and entropy of each feature.
    foreach my $f (@{$data->{features}})
    {
        $fentropy{$f} = 0;
        next if($fcount{$f}==0);
        foreach my $fv (sort(keys(%{$fvcount{$f}})))
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
        my @fvalues = sort(keys(%{$cooc{$f}}));
        foreach my $fv (@fvalues)
        {
            foreach my $g (sort(keys(%{$cooc{$f}{$fv}})))
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
                my @gvalues = sort(keys(%{$cooc{$f}{$fv}{$g}}));
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
    foreach my $f (sort(keys(%centropy)))
    {
        foreach my $g (sort(keys(%{$centropy{$f}})))
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
            # In an attempt to stabilize the results of two consecutive runs
            # (where one of the differences I have observed is a slight difference
            # in mutual information between the same two features), reduce the
            # precision of mutual information. (The numbers I see typically have
            # about 14 decimal digits, let's now limit it to 10.)
            $information{$f}{$g} = $information{$g}{$f} = sprintf("%.10f", $information{$f}{$g});
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
# Modifications of input features.
#==============================================================================



#------------------------------------------------------------------------------
# Modifies input features in a way that might hopefully be more useful for
# predictions. The global configuration hash controls what features will be
# modified and how. This function should be called after the first lh hash has
# been created for the data, so that we can operate on the hash rather than the
# initial table, but before any further information is inferred from the hash.
#------------------------------------------------------------------------------
sub modify_features
{
    my $data = shift;
    # Remember how to restore original values before modification. We may want
    # to do it before we write the data to a CSV file.
    my %restore;
    $data->{restore} = \%restore;
    # Countrycodes == US is unreliable. It occurs with many languages,
    # including e.g. African. Replace it by 'nan'.
    if($config{countrycodes} eq 'nous')
    {
        foreach my $lcode (@{$data->{lcodes}})
        {
            if($data->{lh}{$lcode}{countrycodes} eq 'US')
            {
                $restore{$lcode}{countrycodes} = 'US';
                $data->{lh}{$lcode}{countrycodes} = 'nan';
            }
        }
    }
    elsif($config{countrycodes} ne '')
    {
        die("Unknown parameter countrycodes='$config{countrycodes}'");
    }
    # Latitude and longitude is hardly useful for inference as a single pair
    # of coordinates. However, we can define zones in which we hope languages
    # will share some features.
    if($config{latlon} eq 'zones')
    {
        my @lat =
        (
            90, # North Pole
            60, # 60-90 North (Scandinavia, Northern Russia (excl Kamchatka), most of Alaska (excl Aleutes), Yukon, Northwest Terr, Nunavut, Greenland)
            35, # 35-60 North (Europe and Northernmost Maghreb, Caucasus, Central Asia, Northern China, Korea, most of Japan, most of USA and Canada)
            10, # 10-35 North (Sahara and Sahel, North Ethiopia, Middle East, Most of India and Indochina, Northern Philippines, Southern USA, Mexico, Nicaragua, Caribbean)
            -10, # 10 South - 10 North (Gulf of Guinea, Congo, Tanzania, Sri Lanka, Indonesia, Papua New Guinea, North Peru and Brazil)
            -90 # 10-90 South (South Africa from Angola, Zambia and Mosambique, Madagascar, Australia, New Zealand, South America from Southern Peru, Bolivia and Southern half of Brazil)
        );
        my @lon =
        (
            160,
            -140, # 160 East - 140 West (New Zealand, Solomon Islands, East Kamchatka, Chukotka, Alaska, Hawaii)
            -115, # 115-140 West (Yukon, British Columbia, West Coast of US)
            -95,  # 95-115 West (Northwest Territories, Western Nunavut, Central Canada, Central USA (incl most of Kansas, Texas), Mexico except Yucatan)
            -60,  # 60-95 West (East Canada except Newfoundland, Eastern USA, Caribbean, Amazonia and Andes, most of Bolivia, most of Argentina)
            -25,  # 25-60 West (most of Greenland, Guyana, Eastern Brazil, Paraguay, Uruguay, Northeastern Argentina)
            35,   # 25 West - 35 East (Europe Iceland to almost Moscow, Crimea, Western Turkey, Southwestern Israel, Africa except East Coast from Sudan down)
            70,   # 35-70 East (rest of European Russia, West Kazakhstan and Uzbekistan, middle East and most Afghanistan, Western Pakistan, East Coast of Africa, Madagascar)
            95,   # 70-95 East (Western Siberia, Xinjiang, Tibet, India, Bangladesh, Western Myanmar, Sri Lanka)
            118,  # 95-118 East (Eastern Siberia, Mongolia, Central and Southeastern China, Indochina, most of Indonesia (incl Kalimantan, Java, Bali), West Coast of Australia)
            130,  # 118-130 East (West border of Yakutia, Amur Region, Eastern China, Korea, Taiwan, Philippines, Sulawesi, Western Australia)
            160   # 130-160 East (Yakutia, Southwestern Kamchatka, Japan, Papua New Guinea, Central and Eastern Australia)
        );
        # For some specifically picked regions, register also directly their bounding box.
        my @latlon =
        (
            # minlat, maxlat, minlon, maxlon, region
            #[ 42,  60, -141, -114, 'pacusacan'], # Pacific USA and Canada
            #[ 21,  42, -125,  -93, 'usamexico'], # Southwestern USA and Northern Mexico
            [ 13,  21, -106,  -86, 'guatemala'], # Southern Mexico and Guatemala
            #[ -7,   4,  -75,  -60, 'wamazonia'], # Western Amazonia
            [  2,  18,  -17,   17, 'sahel'    ], # Western Sahel ###!!! First box that somewhat helps!
            [ 34,  47,   17,   29, 'balkans'  ], # Balkans
            #[ 41,  45,   38,   49, 'caucasus' ], # Caucasus
            [  5,  26,   66,   91, 'india'    ], # India, Sri Lanka, Bangladesh but not Nepal and not easternmost India
            #[  9,  37,   99,  123, 'indochina'], # Indochina, Southeastern China
            #[-11,   0,  131,  151, 'newguinea'], # New Guinea
            [-44, -11,  112,  154, 'australia'], # Australia
            #[-50,  30,  163, -120, 'oceania'  ], # Oceania
        );
        foreach my $lcode (@{$data->{lcodes}})
        {
            for(my $i = 0; $i < $#lat; $i++)
            {
                if($data->{lh}{$lcode}{latitude} >= $lat[$i+1])
                {
                    $restore{$lcode}{latitude} = $data->{lh}{$lcode}{latitude};
                    $data->{lh}{$lcode}{latitude} = $lat[$i+1].'–'.$lat[$i];
                    last;
                }
            }
            for(my $i = 0; $i < $#lon; $i++)
            {
                if($lon[$i] > $lon[$i+1] && ($data->{lh}{$lcode}{longitude} > $lon[$i] || $data->{lh}{$lcode}{longitude} <= $lon[$i+1]) ||
                   $lon[$i] < $lon[$i+1] && $data->{lh}{$lcode}{longitude} <= $lon[$i+1])
                {
                    $restore{$lcode}{longitude} = $data->{lh}{$lcode}{longitude};
                    $data->{lh}{$lcode}{longitude} = $lon[$i].'–'.$lon[$i+1];
                    last;
                }
            }
            # In addition to grouping coordinates to zones, we may add a new feature
            # which combines the latitude and longitude zones into 2D areas.
            if($config{latlon2d})
            {
                my $zone2d;
                # First try predefined regions. If none of them matches, use a
                # combination of latitude zone and longitude zone.
                # We must look at $restore{$lcode} because the coordinates in $data->{lh}{$lcode} have been zoned already!
                foreach my $box (@latlon)
                {
                    if($restore{$lcode}{latitude} >= $box->[0] &&
                       $restore{$lcode}{latitude} <= $box->[1] &&
                       ($box->[2] <= $box->[3] &&
                        $restore{$lcode}{longitude} >= $box->[2] &&
                        $restore{$lcode}{longitude} <= $box->[3]) ||
                       ($box->[2] > $box->[3] &&
                        ($restore{$lcode}{longitude} >= $box->[2] ||
                         $restore{$lcode}{longitude} <= $box->[3])))
                    {
                        $zone2d = $box->[4];
                        #print STDERR ("A $data->{lh}{$lcode}{family} language located in $box->[4].\n");
                        last;
                    }
                }
                if(!defined($zone2d))
                {
                    $zone2d = $data->{lh}{$lcode}{latitude}.';'.$data->{lh}{$lcode}{longitude};
                }
                $data->{lh}{$lcode}{latlon} = $zone2d;
            }
        }
        if($config{latlon2d})
        {
            push(@{$data->{features}}, 'latlon');
        }
    }
    elsif($config{latlon} ne '')
    {
        die("Unknown parameter latlon='$config{latlon}'");
    }
    # Create a combined feature from each pair of non-empty features.
    if($config{feats2d})
    {
        my %features2d;
        foreach my $lcode (@{$data->{lcodes}})
        {
            my @nefeatures = grep {exists($data->{lh}{$lcode}{$_}) && $data->{lh}{$lcode}{$_} ne '' && $data->{lh}{$lcode}{$_} !~ m/^nan|\?$/} (@{$data->{features}});
            foreach my $f (@nefeatures)
            {
                foreach my $g (@nefeatures)
                {
                    if($f lt $g)
                    {
                        my $fg = $f.';'.$g;
                        my $fgv = $data->{lh}{$lcode}{$f}.';'.$data->{lh}{$lcode}{$g};
                        $features2d{$fg}++;
                        $data->{lh}{$lcode}{$fg} = $fgv;
                    }
                }
            }
        }
        # We are facing combinatorial explosion. In WALS, one language may have up to 193 features.
        # Non-empty combinations shoot us to 18691 features. But then we will look of co-occurrences
        # of pairs of features, i.e., square the number again, that's ~ 350 million co-occurrences,
        # which is not feasible. Therefore, we will only look at feature pairs that have been
        # observed enough times.
        ###!!! We want to do this cut for training data but not for blind test data!
        ###!!! Hack: If the total number is over 18000, we will assume that we are working with training data.
        my @features2d = sort(keys(%features2d));
        if(scalar(@features2d)>18000)
        {
            @features2d = grep {$features2d{$_}>1000} (@features2d);
        }
        push(@{$data->{features}}, @features2d);
        my $n = scalar(@{$data->{features}});
        print STDERR ("We have $n features after expansion.\n");
    }
}



#==============================================================================
# Statistics and sanity checks.
#==============================================================================



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
    my @languages = @{$data->{lcodes}};
    my $nnan = 0;
    my $nqm = 0;
    my $nreg = 0;
    foreach my $l (@languages)
    {
        foreach my $feature (@{$data->{features}})
        {
            my $value = $data->{lh}{$l}{$feature};
            if($value eq 'nan')
            {
                $nnan++;
            }
            elsif($value eq '?')
            {
                $nqm++;
            }
            else
            {
                $nreg++;
            }
        }
    }
    print STDERR ("Found $nnan nan values.\n");
    print STDERR ("Found $nqm ? values to be predicted.\n");
    print STDERR ("Found $nreg regular non-empty values.\n");
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
            $minreg_langname = $data->{lh}{$l}{name};
        }
    }
    my $avgqm = $sumqm/$nl;
    my $avgreg = $sumreg/$nl;
    print STDERR ("On average, a language has $avgreg non-empty features and $avgqm features to predict.\n");
    print STDERR ("Minimum knowledge is $minreg non-empty features (language $minreg_langname); in that case, $minreg_qm features are to be predicted.\n");
    print STDERR ("Note that the non-empty features always include 8 non-typologic features: ord, code, name, latitude, longitude, genus, family, countrycodes.\n");
}



#------------------------------------------------------------------------------
# Compares two data sets that have been just read (i.e., %data does not yet
# contain hashed indexes). Looks for features that exist in one data set and
# not in the other.
#------------------------------------------------------------------------------
sub compare_data_sets
{
    my $d1 = shift; # hash ref
    my $d2 = shift; # hash ref
    my $wals = shift; # optional hash ref (if defined, then it should have item loaded=1)
    my $f1 = $d1->{features};
    my $f2 = $d2->{features};
    my %hf1;
    foreach my $feature (@{$f1})
    {
        $hf1{$feature}++;
    }
    my %hf2;
    foreach my $feature (@{$f2})
    {
        $hf2{$feature}++;
        if(!exists($hf1{$feature}))
        {
            print STDERR ("Feature '$feature' exists only in the second dataset.\n");
        }
    }
    foreach my $feature (@{$f1})
    {
        if(!exists($hf2{$feature}))
        {
            print STDERR ("Feature '$feature' exists only in the first dataset.\n");
        }
    }
    # If the first set is for training and the second for dev or test, then we
    # do not expect a language to appear in both sets.
    my $l1 = $d1->{lcodes};
    my $l2 = $d2->{lcodes};
    my %hl1;
    foreach my $lcode (@{$l1})
    {
        $hl1{$lcode}++;
    }
    foreach my $lcode (@{$l2})
    {
        if(exists($hl1{$lcode}))
        {
            print STDERR ("Language '$lcode' occurs in both sets.\n");
        }
    }
    # If we have access to the current WALS data, check that the feature names are known.
    if($wals->{loaded})
    {
        print STDERR ("Checking data against WALS 2020...\n");
        # Invert the hash of parameters (features) so that we can search feature names.
        my %wfeatures;
        foreach $pid (keys(%{$wals->{parameters}}))
        {
            my $name = $wals->{parameters}{$pid}{name};
            # Spaces are replaced with underscores in the SIGTYP shared task.
            $name =~ s/\s+/_/g;
            $wfeatures{$name} = $wals->{parameters}{$pid};
        }
        # Check that all features in d1 are known in WALS.
        foreach my $feature (@{$f1})
        {
            if(!exists($wfeatures{$feature}))
            {
                print STDERR ("Feature '$feature' from the first dataset is not know in WALS.\n");
            }
        }
        # Check that all features in d2 are known in WALS.
        foreach my $feature (@{$f2})
        {
            if(!exists($wfeatures{$feature}))
            {
                print STDERR ("Feature '$feature' from the second dataset is not know in WALS.\n");
            }
        }
    }
}
