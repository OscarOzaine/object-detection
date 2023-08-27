# -*- coding: utf-8 -*
from wrappers import *
from utils import compute_centers

def download_gmaps_api(places, folder='../../data'):
    GMAPS_KEY = os.environ.get('GMAPS_KEY')

    gmaps = GoogleMapsAPIDownloader(GMAPS_KEY)

    for name, coords in places.items():

        path = f'{folder}/{name}'
        centers = compute_centers(*coords)
        print(f'Number of tiles: {len(centers)*len(centers[0])} ({len(centers)}x{len(centers[0])})')

        gmaps.parallel_download_grid(
            centers, 
            path,
            split=False,
            maptype='satellite',
            format='png',
            size=(1280, 1280),
            zoom=19, 
            scale=2,
        )


def download_gmaps_web(places, folder='../../data'):
    gmaps = GoogleMapsWebDownloader()

    for name, coords in places.items():

        path = f'{folder}/{name}'
        gmaps.download(
            *coords, folder=path,
            zoom=19, style='s', format='png'
        )

        # print('Merging tiles...')
        # gmaps.merge(f'{folder}/merged.png')


def main():
    folder = 'data'
    PLACES = {
        # 'yuma': [(32.7048772,-114.6481852), (32.7087466,-114.6369637)],
        'yuma': [(32.6881707,-114.6660823), (32.7232069,-114.6182909)],
    }
    
    download_gmaps_api(PLACES, folder)

    # start_time = time.time()
    # download_gmaps_web(PLACES, folder)

    # final_time = time.time() - start_time
    # total_files = len(os.listdir(folder))
    # print(f'\nDownloaded files: {total_files}')
    # print(f'{total_files / final_time} files/second')
    # print(f'Elapsed time: {final_time}s')


if __name__ == '__main__':
    main()
