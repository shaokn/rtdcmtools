#!/usr/bin/env python3
"""Anonymize DICOM files while preserving the source directory structure."""

import argparse
from pathlib import Path

import pydicom


def extract_initials(name):
    words = name.split()
    initials = [word[0].upper() for word in words]
    return ''.join(initials)


def anonymize_dicom_file(dicom_path, output_path, new_prefix=''):
    dataset = pydicom.dcmread(dicom_path)

    if 'PatientID' in dataset:
        old_id = dataset.data_element('PatientID').value
        dataset.data_element('PatientID').value = new_prefix + old_id

    if 'PatientName' in dataset:
        old_name = str(dataset.data_element('PatientName').value)
        dataset.data_element('PatientName').value = 'Anonymized' + extract_initials(old_name)

    if 'InstitutionName' in dataset:
        dataset.data_element('InstitutionName').value = ''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(output_path)


def anonymize_dicom_directory(input_directory, output_directory, new_prefix=''):
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    files = sorted(input_directory.rglob('*.dcm'))
    for dicom_path in files:
        relative_path = dicom_path.relative_to(input_directory)
        anonymize_dicom_file(dicom_path, output_directory / relative_path, new_prefix)
    return len(files)


def main():
    parser = argparse.ArgumentParser(
        description='匿名化DICOM并保留输入目录的相对结构。仅修改PatientID、PatientName、InstitutionName。')
    parser.add_argument('input_directory', type=Path)
    parser.add_argument('output_directory', type=Path)
    parser.add_argument('new_prefix', nargs='?', default='Breast')
    args = parser.parse_args()

    print(f'输入: {args.input_directory}')
    print(f'输出: {args.output_directory}')
    print(f'前缀: {args.new_prefix}')
    count = anonymize_dicom_directory(args.input_directory, args.output_directory, args.new_prefix)
    print(f'Done! Anonymized {count} DICOM files.')


if __name__ == '__main__':
    main()
