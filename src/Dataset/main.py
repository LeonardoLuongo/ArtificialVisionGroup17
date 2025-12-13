


def donwnload_tracking_2023():
    from SoccerNet.Downloader import SoccerNetDownloader
    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="./dataset")
    mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])



if __name__ == "__main__":
    donwnload_tracking_2023()