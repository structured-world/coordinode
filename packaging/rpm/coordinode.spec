# CoordiNode RPM spec.
#
# Build expects a pre-compiled musl-static `coordinode` binary in SOURCES/.
# The CI release pipeline runs `cargo build --release --target $TARGET-unknown-linux-musl --bin coordinode`,
# strips the result, copies it (and the packaging assets) into the rpmbuild
# tree, then invokes `rpmbuild -bb`.

%global _hardened_build 1
%global debug_package %{nil}

Name:           coordinode
Version:        %{?version}%{!?version:0.0.0}
Release:        1%{?dist}
Summary:        Unified database for AI-native applications

License:        AGPL-3.0-only
URL:            https://github.com/structured-world/coordinode

Source0:        coordinode
Source1:        coordinode.service
Source2:        coordinode.conf
Source3:        coordinode.sysusers
Source4:        coordinode.tmpfiles
Source5:        LICENSE
Source6:        README.md

BuildRequires:  systemd-rpm-macros
%{?systemd_requires}

Requires(pre):  shadow-utils
Requires:       systemd

# Match the architectures we publish.
ExclusiveArch:  x86_64 aarch64

%description
CoordiNode — unified database for AI-native applications.
Graph + Vector + Full-Text + Document + Time-Series in one engine,
one query language (OpenCypher), one transaction.

# ── Build phases ────────────────────────────────────────────────────
# No %prep / %build: Source0 is already the final binary produced by
# the CI musl toolchain. We only stage it into the buildroot.

%install
install -D -m 0755 %{SOURCE0}                              %{buildroot}%{_bindir}/coordinode
install -D -m 0644 %{SOURCE1}                              %{buildroot}%{_unitdir}/coordinode.service
install -D -m 0644 %{SOURCE2}                              %{buildroot}%{_sysconfdir}/coordinode/coordinode.conf
install -D -m 0644 %{SOURCE3}                              %{buildroot}%{_sysusersdir}/coordinode.conf
install -D -m 0644 %{SOURCE4}                              %{buildroot}%{_tmpfilesdir}/coordinode.conf
install -D -m 0644 %{SOURCE5}                              %{buildroot}%{_licensedir}/%{name}/LICENSE
install -D -m 0644 %{SOURCE6}                              %{buildroot}%{_docdir}/%{name}/README.md

# Note: /var/lib/coordinode is intentionally NOT packaged. systemd-tmpfiles
# creates it (per coordinode-tmpfiles.conf) on install/boot, and `rpm -e`
# leaves it untouched — critical for a database: a stray uninstall must never
# wipe persistent state. Admins must remove it manually if they want a clean
# slate.

# ── Scriptlets ──────────────────────────────────────────────────────

%pre
# Materialise the system user immediately from the sysusers.d snippet so the
# %files ownership directives below resolve on first install (RPM applies
# attrs after %pre and after %post; sysusers.d processing fires from %post
# via macros, too late for ownership). Idempotent on upgrades.
%sysusers_create_compat %{SOURCE3}

%post
%systemd_post coordinode.service
# Re-apply tmpfiles.d in case the admin already ran systemctl daemon-reload
# without rebooting (covers fresh install on a long-running host).
%tmpfiles_create %{_tmpfilesdir}/coordinode.conf || :

%preun
%systemd_preun coordinode.service

%postun
%systemd_postun_with_restart coordinode.service

# ── Payload ─────────────────────────────────────────────────────────
%files
%license %{_licensedir}/%{name}/LICENSE
%doc %{_docdir}/%{name}/README.md
%{_bindir}/coordinode
%{_unitdir}/coordinode.service
%{_sysusersdir}/coordinode.conf
%{_tmpfilesdir}/coordinode.conf
%dir %{_sysconfdir}/coordinode
%config(noreplace) %{_sysconfdir}/coordinode/coordinode.conf

%changelog
* %(date "+%a %b %d %Y") Release Bot <oss@sw.foundation> - %{version}-1
- Automated release; see CHANGELOG.md for details.
