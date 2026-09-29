$out_dir = 'build';
$aux_dir = 'build';
$pdf_mode = 1;
$jobname = 'Giuseppe_Gabriele_russo_Master_thesis';
$success_cmd = 'perl scripts/create_pdfa.pl';

# The macOS universal Biber launcher can fail while extracting its native binary.
# Extract that same installed binary into the ignored build directory when needed.
if ($^O eq 'darwin') {
    require File::Spec;
    my ($installed_biber) = grep { -x $_ }
        map { File::Spec->catfile($_, 'biber') } File::Spec->path();
    my $arch = `uname -m`;
    chomp $arch;
    if ($installed_biber && $arch eq 'arm64') {
        my $native = File::Spec->rel2abs('build/biber-arm64');
        mkdir 'build' unless -d 'build';
        if (!-f $native || (stat($native))[9] < (stat($installed_biber))[9]) {
            system('/usr/bin/lipo', $installed_biber, '-thin', 'arm64', '-output', $native) == 0
                or die "Cannot prepare native Biber from $installed_biber\n";
        }
        $biber = qq{"$native" %O %S};
    }
}
