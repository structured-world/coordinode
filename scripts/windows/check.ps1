# The CI gate for one bundled tree, run on a Windows machine by check.sh.
#
# Usage: check.ps1 -Root <work dir> -Bundle <bundle path> -Ref <ref in bundle>
#                  [-Nextest <extra nextest arguments>]
# Writes status.txt and one log per step into -Root, then removes the checkout
# and the build output: the machine is shared, nothing stays. With -Nextest,
# only nextest runs, with those arguments added, split at whitespace.
param(
    [Parameter(Mandatory)] [string] $Root,
    [Parameter(Mandatory)] [string] $Bundle,
    [Parameter(Mandatory)] [string] $Ref,
    [string] $Nextest = ''
)

# Keep the machine awake while this process runs (ES_CONTINUOUS |
# ES_SYSTEM_REQUIRED); the flag dies with the process.
Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null

# The idle timer keeps running under the hold, so by the end of a run it has
# run out, and the machine sleeps the moment the hold goes, before check.sh
# fetches the logs; resetting the timer does not stop it after an unattended
# wake. So the run says it is done and keeps the hold until check.sh creates
# the file `fetched` beside the logs, or for ten minutes at most. (A script
# run with -File over ssh does not see the session's standard input.)
function Wait-Fetched {
    $marker = Join-Path $Root 'fetched'
    $watcher = New-Object IO.FileSystemWatcher $Root, 'fetched'
    [Console]::Out.WriteLine('CHECK-DONE')
    [Console]::Out.Flush()
    if (-not (Test-Path $marker)) {
        $watcher.WaitForChanged([IO.WatcherChangeTypes]::Created, 600000) | Out-Null
    }
    $watcher.Dispose()
}

# Windows PowerShell writes UTF-16 by default; check.sh reads status.txt as
# plain text.
$PSDefaultParameterValues['Out-File:Encoding'] = 'ascii'

$src = Join-Path $Root 'src'
$target = Join-Path $Root 'target'
$status = Join-Path $Root 'status.txt'

Remove-Item $status -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $src -ErrorAction SilentlyContinue

git clone -q --no-checkout $Bundle $src
Set-Location $src
# A clone takes branches only; the snapshot lives on its own ref.
git fetch -q $Bundle $Ref
git checkout -q --detach FETCH_HEAD
# Each submodule comes from the bundle check.sh sent with the tree
# (sub-<name>.bundle), which holds commits its upstream may not have yet.
git submodule init -q
Get-ChildItem $Root -Filter 'sub-*.bundle' | ForEach-Object {
    $name = $_.BaseName.Substring(4)
    git config "submodule.$name.url" $_.FullName
}
git -c protocol.file.allow=always submodule update -q
"checkout=$LASTEXITCODE" | Out-File $status
if ($LASTEXITCODE -ne 0) {
    Set-Location $Root
    Remove-Item -Recurse -Force $src, $Bundle, (Join-Path $Root 'sub-*.bundle') -ErrorAction SilentlyContinue
    'done' | Out-File -Append $status
    Wait-Fetched
    exit 1
}

# The C runtime is linked statically, as a product embedding the library
# links it; the same flags as the CI job.
$env:RUSTFLAGS = '-D warnings -C target-feature=+crt-static'
$env:COORDINODE_TEST_RAFT_GENEROUS_TIMEOUTS = '1'
$env:CARGO_TARGET_DIR = $target

if ($Nextest) {
    $extra = $Nextest -split '\s+' | Where-Object { $_ }
    $log = Join-Path $Root 'test.log'
    cargo nextest run --all-features --workspace --no-fail-fast --failure-output final @extra *> $log
    $code = $LASTEXITCODE
    # A stress run (--stress-count) exits 0 when some of its iterations
    # failed (cargo-nextest 0.9.146); its summary line still counts them.
    if ($code -eq 0 -and (Select-String -Path $log -Pattern '^\s*Summary .* [1-9][0-9]* failed' -Quiet)) { $code = 1 }
    "nextest=$code" | Out-File -Append $status
    Set-Location $Root
    Remove-Item -Recurse -Force $src, $target, $Bundle -ErrorAction SilentlyContinue
    'done' | Out-File -Append $status
    Wait-Fetched
    exit 0
}

cargo clippy --workspace --all-targets --all-features -- -D warnings *> (Join-Path $Root 'clippy.log')
"clippy=$LASTEXITCODE" | Out-File -Append $status
# The integration tests run the server binary.
cargo build --all-features -p coordinode-server *> (Join-Path $Root 'build.log')
"build=$LASTEXITCODE" | Out-File -Append $status
cargo nextest run --all-features --workspace --no-fail-fast --status-level fail --final-status-level fail --failure-output final *> (Join-Path $Root 'test.log')
"nextest=$LASTEXITCODE" | Out-File -Append $status
cargo test --doc --all-features *> (Join-Path $Root 'doctest.log')
"doctest=$LASTEXITCODE" | Out-File -Append $status

Set-Location $Root
Remove-Item -Recurse -Force $src, $target, $Bundle -ErrorAction SilentlyContinue
'done' | Out-File -Append $status
Wait-Fetched
