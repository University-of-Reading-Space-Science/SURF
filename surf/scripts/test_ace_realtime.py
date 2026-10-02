"""Quick manual test of the NOAA ACE real-time import and plotting."""

import argparse
import datetime

import matplotlib.pyplot as plt

from surf.surf_insitu import get_ace_realtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', help='UTC start time, e.g. 2026-07-01T00:00')
    parser.add_argument('--end', help='UTC end time, e.g. 2026-07-03T00:00')
    args = parser.parse_args()

    end = (datetime.datetime.fromisoformat(args.end) if args.end else
           datetime.datetime.now(datetime.timezone.utc).replace(
               minute=0, second=0, microsecond=0) - datetime.timedelta(days=50))
    start = (datetime.datetime.fromisoformat(args.start) if args.start else
             end - datetime.timedelta(days=27))

    ace = get_ace_realtime(start, end)
    print(ace.head())
    print(f'Imported {len(ace)} records from {ace.datetime.min()} to '
          f'{ace.datetime.max()}')

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
    axes[0].plot(ace['datetime'], ace['V'])
    axes[0].set_ylabel('V [km/s]')
    axes[1].plot(ace['datetime'], ace['N'])
    axes[1].set_ylabel('N [cm$^{-3}$]')
    axes[2].plot(ace['datetime'], ace['T'])
    axes[2].set_ylabel('T [K]')
    axes[3].plot(ace['datetime'], ace['BX_GSE'], label='Bx')
    axes[3].plot(ace['datetime'], ace['BZ_GSM'], label='Bz')
    axes[3].set_ylabel('B [nT]')
    axes[3].set_xlabel('UTC')
    axes[3].legend()

    fig.suptitle('ACE real-time solar wind')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
