#!/usr/bin/env perl
# Reads blind test data in SIGTYP format and predicted data in ÚFAL format.
# Completes the features in the blind data and prints it to STDOUT, that is,
# in the SIGTYP format and making sure that the order of the features matches
# the input.
# WARNING:
# Liz actually said that we should submit the data in the format of the trial
# data (https://github.com/sigtyp/ST2020/blob/master/data/trial/test_trial_data.tab),
# which is different from what they gave us in test_blinded.csv. Some columns
# are missing (genus, latitude, longitude, countrycodes). Feature names seem to
# be lowercased. And what about the TABs in feature values? And should it contain
# only the predicted features, or all?
# Copyright © 2020 Dan Zeman <zeman@ufal.mff.cuni.cz>
# License: GNU GPL

use utf8;
use open ':utf8';
binmode(STDIN, ':utf8');
binmode(STDOUT, ':utf8');
binmode(STDERR, ':utf8');
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

sub usage
{
    print STDERR ("Usage: perl scripts/convert_prediction_to_sigtyp.pl data/test_blinded.csv outputs/my_output-test.csv > ÚFAL_constrained.tab\n");
}

my $n = scalar(@ARGV);
if($n != 2)
{
    usage();
    die("Expected 2 arguments, found $n");
}
# Read the predicted data and store it in a hash so that the feature values can be accessed easily.
my %predicted = Sigtypio::read_csv($ARGV[1]);
Sigtypio::convert_table_to_lh(\%predicted, 0);
# Read the blind data, fill in the missing feature values and write it again.
open(BLIND, $ARGV[0]) or die("Cannot read '$ARGV[0]': $!");
# Read the header line and write our own header line (tab-separated; the input is two-spaces-separated).
my $headers = <BLIND>;
if($headers !~ m/^wals_code/)
{
    print STDERR ("WARNING: The first line of the blind data does not look like a header line:\n");
    print STDERR ("         $headers\n");
}
print("wals_code\tname\tfamily\tfeatures\n");
my $n_predicted = 0;
while(<BLIND>)
{
    my $line = $_;
    $line =~ s/\r?\n$//;
    my @f = split(/\t/, $line);
    # The blind data normally contains 8 tab-separated columns: wals_code, name, latitude, longitude, genus, family, countrycodes, features.
    # If it contains more than 8 columns, it means that one or more feature values contain the TAB character. Join them.
    if(scalar(@f) < 8)
    {
        print STDERR ("WARNING: Blind data line contains fewer than 8 tab-separated columns:\n");
        print STDERR ("$line\n");
    }
    elsif(scalar(@f) > 8)
    {
        my $features = join('', @f[7..$#f]);
        splice(@f, 7, scalar(@f)-7, $features);
    }
    my $lcode = $f[0];
    my $lname = $f[1];
    my $family = $f[5];
    my $features = $f[7];
    my @features = split(/\|/, $features);
    foreach my $fv (@features)
    {
        if($fv =~ m/^([^=]+)=(.+)$/)
        {
            my $f = $1;
            my $v = $2;
            if($v eq '?')
            {
                if(exists($predicted{lh}{$lcode}{$f}))
                {
                    $vp = $predicted{lh}{$lcode}{$f};
                    if(!defined($vp) || $vp eq '' || $vp eq 'nan')
                    {
                        print STDERR ("WARNING: Undefined or empty prediction for language '$lname' feature '$f'\n");
                    }
                    else
                    {
                        $v = $vp;
                        $n_predicted++;
                    }
                }
                else
                {
                    print STDERR ("WARNING: No available prediction for language '$lname' feature '$f'\n");
                }
            }
            # In the trial test data, feature names are lowercased.
            $fv = lc($f).'='.$v;
        }
        else
        {
            print STDERR ("WARNING: Cannot separate feature name from value in '$fv'\n");
        }
    }
    $features = join('|', @features);
    print("$lcode\t$lname\t$family\t$features\n");
}
close(BLIND);
print STDERR ("Predicted $n_predicted feature values in total.\n");