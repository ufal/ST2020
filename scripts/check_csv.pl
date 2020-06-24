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
print STDERR ("Found $data{nf} headers.\n");
print STDERR ("Found $data{nl} language lines.\n");
Sigtypio::convert_table_to_lh(\%data, 0);
if($write)
{
    Sigtypio::write_csv(\%data);
}
