"""Command-line contract of server_new.py: bind address and startup banner.

The banner is what a user copies into a browser, so these tests pin down two
things: the bind address actually reaching Flask, and the fact that 0.0.0.0 is
never advertised as a browsing address.
"""
import contextlib
import io
import sys
import unittest
from unittest.mock import patch

import server_new


class LoopbackTests(unittest.TestCase):
    def test_loopback_hosts(self):
        for host in ('', 'localhost', '127.0.0.1', '127.0.1.1', '::1'):
            self.assertTrue(server_new.is_loopback_host(host), host)

    def test_non_loopback_hosts(self):
        for host in ('0.0.0.0', '::', '10.123.253.36', '192.168.1.7'):
            self.assertFalse(server_new.is_loopback_host(host), host)


class BannerTests(unittest.TestCase):
    def test_default_host_advertises_loopback_only(self):
        lines = server_new.startup_lines('127.0.0.1', 8768, 'dicom')
        self.assertEqual(lines, ['New viewer: http://127.0.0.1:8768/?source=dicom'])
        self.assertNotIn('0.0.0.0', '\n'.join(lines))

    def test_any_host_expands_into_real_addresses(self):
        with patch.object(server_new, 'lan_ipv4_addresses', return_value=['10.0.0.5']):
            lines = server_new.startup_lines('0.0.0.0', 7777, 'dicom')
        joined = '\n'.join(lines)
        self.assertIn('http://127.0.0.1:7777/?source=dicom', joined)
        self.assertIn('http://10.0.0.5:7777/?source=dicom', joined)
        # 0.0.0.0 is a bind keyword; printing it as a URL would send the user nowhere.
        self.assertNotIn('http://0.0.0.0', joined)
        self.assertIn('no authentication', joined)

    def test_any_host_without_a_detectable_address_still_guides_the_user(self):
        with patch.object(server_new, 'lan_ipv4_addresses', return_value=[]):
            lines = server_new.startup_lines('0.0.0.0', 7777, 'nifti')
        joined = '\n'.join(lines)
        self.assertIn('<this machine', joined)
        self.assertIn('?source=nifti', joined)

    def test_single_interface_is_reported_verbatim(self):
        with patch.object(server_new, 'lan_ipv4_addresses', return_value=['10.0.0.5']):
            lines = server_new.startup_lines('10.0.0.9', 7777, 'dicom')
        joined = '\n'.join(lines)
        self.assertEqual(len([line for line in lines if 'New viewer' in line]), 1)
        self.assertIn('http://10.0.0.9:7777/?source=dicom', joined)

    def test_lan_ipv4_addresses_never_return_loopback_or_any_host(self):
        for address in server_new.lan_ipv4_addresses():
            self.assertFalse(server_new.is_loopback_host(address), address)
            self.assertNotEqual(address, '0.0.0.0')


class MainArgumentTests(unittest.TestCase):
    def setUp(self):
        # main() rebinds the module-level data roots to the empty temporary
        # session, which the tests that read the workstation case data rely on.
        # Snapshot them and put them back once the test is done.
        self.original = (server_new.core.ROOT, server_new.core.NIFTI_ROOT,
                         server_new.core.app.config.get('DEFAULT_SOURCE'))
        self.addCleanup(self.restore_globals)

    def restore_globals(self):
        root, nifti_root, source = self.original
        server_new.core.ROOT = root
        server_new.core.NIFTI_ROOT = nifti_root
        server_new.core.app.config['DEFAULT_SOURCE'] = source

    def run_main(self, argv):
        """Run main() with Flask's run() stubbed out, returning (kwargs, output)."""
        buffer = io.StringIO()
        with patch.object(sys, 'argv', ['server_new.py'] + argv), \
                patch.object(server_new.core.app, 'run') as run, \
                contextlib.redirect_stdout(buffer):
            server_new.main()
        return run.call_args.kwargs, buffer.getvalue()

    def test_default_binds_loopback(self):
        kwargs, output = self.run_main([])
        self.assertEqual(kwargs['host'], '127.0.0.1')
        self.assertEqual(kwargs['port'], 8768)
        self.assertTrue(kwargs['threaded'])
        self.assertFalse(kwargs['debug'])
        self.assertIn('http://127.0.0.1:8768/?source=dicom', output)
        self.assertNotIn('no authentication', output)

    def test_host_and_port_are_forwarded(self):
        kwargs, output = self.run_main(['--host', '0.0.0.0', '--port', '7777'])
        self.assertEqual(kwargs['host'], '0.0.0.0')
        self.assertEqual(kwargs['port'], 7777)
        self.assertIn('no authentication', output)

    def test_source_flag_reaches_the_banner(self):
        _, output = self.run_main(['--default-source', 'nifti'])
        self.assertIn('?source=nifti', output)


if __name__ == '__main__':
    unittest.main()
