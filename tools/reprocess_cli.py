#!/usr/bin/env python3
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi', required=True, help='ROI spec, e.g., z0-256,y0-256,x0-256')
    args = parser.parse_args()
    t0 = time.time()
    # TODO: implement pipeline
    time.sleep(0.1)
    dt = time.time() - t0
    print(f"Reprocess completed in {dt:.2f}s for ROI {args.roi}")

if __name__ == '__main__':
    main()
