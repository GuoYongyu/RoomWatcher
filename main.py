# coding: utf-8


if __name__ == '__main__':
    # from tasks.obtain_image import test_run as image_run
    # from tasks.video_record import test_run as video_run

    # image_run()
    # video_run()


    from demo import Demoer
    print("Please wait a moment. The test will run for some time.")
    demoer = Demoer(demo_way="compare-improved", yaml_path="./config/cockroach-detection.yaml")
    demoer.run()
