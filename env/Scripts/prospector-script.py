#!"E:\Major Project\OVAS\env\Scripts\python.exe"
# EASY-INSTALL-ENTRY-SCRIPT: 'prospector==1.2.0','console_scripts','prospector'
__requires__ = 'prospector==1.2.0'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('prospector==1.2.0', 'console_scripts', 'prospector')()
    )
