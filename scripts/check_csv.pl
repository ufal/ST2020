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

my $original = 0; # original data separated by spaces and tabs, with errors; our _x and _y data (original==0) separated by commas
my $write = 0; # we can write the fixed table in our format if desired
GetOptions
(
    'original' => \$original,
    'write'    => \$write
);

my %data = Sigtypio::read_csv();
print STDERR ("Found ", scalar(@{$data{infeatures}}), " headers.\n");
print STDERR ("Found ", scalar(@{$data{table}}), " language lines.\n");
hash_features(\%data, 0);
if($write)
{
    Sigtypio::write_csv(\%data);
}



#------------------------------------------------------------------------------
# Converts the input table to hashes.
#   {lcodes} ...... list of wals codes of known languages (index to lh)
#   {lh} .......... data from {table} indexed by {language}{feature}
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
    my $qm_is_nan = shift; # convert question marks to 'nan'?
    my @lcodes;
    my %lh; # hash indexed by language code
    my %lhclean; # like %lh but only non-empty values
    my %fcount;
    my %fvcount;
    my %fvprob;
    my %fentropy;
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
    # Modify features to improve prediction.
    ###!!!modify_features($data);
    # Now go on and fill the other hashes that depend solely on one language-feature-value triple.
    foreach my $lcode (@lcodes)
    {
        # Remember observed features and values.
        foreach my $feature (@{$data->{features}})
        {
            my $value = $lh{$lcode}{$feature};
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
