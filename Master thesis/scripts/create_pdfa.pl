use strict;
use warnings;
use File::Temp qw(tempdir);
use File::Spec;

# Run from the thesis directory, as latexmk does for the local configuration.
my $base = 'build/Giuseppe_Gabriele_russo_Master_thesis';
my $source = "$base.pdf";
my $target = "${base}_PDFA.pdf";
-f $source or die "Missing compiled thesis: $source\n";

my $gs = $ENV{GHOSTSCRIPT} || 'gs';
my $profile = $ENV{PDFA_ICC_PROFILE};
unless ($profile) {
    my @candidates = (
        glob('/opt/homebrew/share/ghostscript/iccprofiles/srgb.icc'),
        glob('/usr/local/share/ghostscript/*/iccprofiles/srgb.icc'),
        glob('/usr/share/ghostscript/*/iccprofiles/srgb.icc'),
        '/usr/share/color/icc/ghostscript/srgb.icc',
    );
    ($profile) = grep { -f $_ } @candidates;
}
$profile && -f $profile
    or die "Cannot find sRGB ICC profile. Set PDFA_ICC_PROFILE to its path.\n";
$profile = File::Spec->rel2abs($profile);
my $ps_profile = $profile;
$ps_profile =~ s/([\\()])/\\$1/g;
my $tmp = tempdir('pdfa-XXXXXX', DIR => 'build', CLEANUP => 1);
my $definition = "$tmp/definition.ps";
open my $fh, '>', $definition or die "$definition: $!";
print {$fh} <<"POSTSCRIPT";
%!
[/_objdef {icc_PDFA} /type /stream /OBJ pdfmark
[{icc_PDFA} << /N 3 >> /PUT pdfmark
[{icc_PDFA} ($ps_profile) (r) file /PUT pdfmark
[/_objdef {OutputIntent_PDFA} /type /dict /OBJ pdfmark
[{OutputIntent_PDFA} << /Type /OutputIntent /S /GTS_PDFA1
 /DestOutputProfile {icc_PDFA} /OutputConditionIdentifier (sRGB) >> /PUT pdfmark
[{Catalog} << /OutputIntents [{OutputIntent_PDFA}] >> /PUT pdfmark
POSTSCRIPT
close $fh or die "$definition: $!";

# Remove non-printing link annotations that are incompatible with PDF/A.
# Publish only after strict conversion succeeds; preserve the previous output on failure.
my $status = system($gs, '-q', '-dBATCH', '-dNOPAUSE', '-dPDFA=2',
    '-dPDFACompatibilityPolicy=2', '-sDEVICE=pdfwrite',
    '-sColorConversionStrategy=RGB', '-dEmbedAllFonts=true',
    '-dPreserveAnnots=false', '-dAutoRotatePages=/None',
    "--permit-file-read=$profile", "-sOutputFile=$tmp/output.pdf",
    $definition, $source);
$status == 0 or die "PDF/A conversion failed (status $status).\n";
rename "$tmp/output.pdf", $target or die "Cannot publish $target: $!";
print "Created PDF/A-2b: $target\n";
