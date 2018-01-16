import sys
import os
import csv
from nltk.tokenize import RegexpTokenizer

MODE = 0  # Use hard labels
# MODE = 1  # Use soft labels

tar_lookup = {"AvoidBlue_Clean.csv": "F & C G ! B",
              "AvoidGreen_Clean.csv": "F & B G ! C",
              "AvoidRed_Clean.csv": "F & B G ! R",
              "AvoidYellow_Clean.csv": "F & C G ! Y",
              "ThroughBlue_Clean.csv": "F & B F C",
              "ThroughGreen_Clean.csv": "F & C F B",
              "EventGreen_Clean.csv": "F C",
              "EventRed_Clean.csv": "F R",
              "BlockBlue_Clean.csv": "F BB",
              "BlockGreen_Clean.csv": "F BC",
              "BlockBlue_Green_Clean.csv": "F & BB F C",
              "BlockGreen_Blue_Clean.csv": "F & BC F B"}


def main(data_files):
    prefix = 'hard' if MODE == 0 else 'soft'
    out_srcf = '{0}_pc_src.txt'.format(prefix)
    out_tarf = '{0}_pc_tar.txt'.format(prefix)
    data_files = data_files
    tokenizer = RegexpTokenizer(r'\w+')

    src_lines, tar_lines = [], []
    for f in data_files:
        total, used = 0, 0
        with open(f, 'rb') as data:
            reader = csv.DictReader(data)
            for line in reader:
                if line['Commands']:
                    total += 1
                    if line['Hard Label' if MODE == 0 else 'Soft Label'].strip() == '1':
                        src_lines.append(' '.join(tokenizer.tokenize(line['Commands'])).lower().strip())
                        tar_lines.append(tar_lookup[os.path.basename(f)])
                        used += 1

            print 'Processed file {0}: added {1}/{2} sentences'.format(f, used, total)

    print 'Writing parallel corpus of {0} sentences to disk...'.format(len(src_lines))
    with open(out_srcf, 'wb') as out_src, open(out_tarf, 'wb') as out_tar:
        for sl, tl in zip(src_lines, tar_lines):
            out_src.write(sl + '\n')
            out_tar.write(tl + '\n')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: python generate_pc.py <data_file1> ... <date_fileN>'
        sys.exit(0)

    main(sys.argv[1:])
