#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <shop.h>
#include <iostream>
#include <getwindow.cpp>
#include <filesystem>


class Timer {
public:

    Timer() {
        m_StartTimepoint = std::chrono::high_resolution_clock::now();

    }

    ~Timer() {
        Stop();

    }

    void Stop() {
        auto endTimepoint = std::chrono::high_resolution_clock::now();

        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimepoint).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();

        auto duration = end - start;
        double ms = duration * 0.001;
        std::cout << "Benchmark: " << ms << "ms\n";

    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;

};

#include <tesseractOcr.h>

int main(int argc, char* argv[]) {
    cv::Mat fullScreen = cv::imread("D:/linux_img_port/Fullscreen_different_contrast.png");
    cv::Mat shopTemplate = cv::imread("D:/linux_img_port/shop_2560x1440.PNG");
    
    Timer* myTime = new Timer;  
    Shop myshop;
    /*
    char start;
    do {
        std::cout << "Ready for main display capture(5s delay) [Y/N]:\n";
        std::cin >> start;
        if(start == 'N' || start == 'n') {
            std::cout << "Exiting...\n";
            return 0;
        }
        else if(start != 'Y' && start != 'y') {
            std::cout << "Invalid input, please enter Y or N.\n";
		}
    } while (start != 'Y' && start != 'y');

    for (int i = 0; i < 5; i++) {
        std::cout << 5 - i << std::endl;
        Sleep(1000);
    }
    
    cv::Mat fullScreen = grabScreenMat();
    
    cv::imwrite("D:/linux_img_port/Shop_class/screen_cap_test.png", fullScreen);
    */
    myshop.locateShopELements(ImageRegion(fullScreen.clone(), cv::Rect(0, 0, fullScreen.cols, fullScreen.rows)), shopTemplate);
    myshop.testElementIRVitals(fullScreen);

	std::vector<std::string> champNames = myshop.getChampNames();

	std::cout << "Champ Names:\n";
    for(const auto& name : champNames) {
        std::cout << name << " ";
	}
	std::cout << std::endl;

	std::cout << "Level: " << myshop.getLevel() << std::endl;
	std::cout << "Level Progress: " << myshop.getLevelProgress() << std::endl;
	std::cout << "Gold: " << myshop.getGold() << std::endl;

    /*
	cv::Mat missingSlotFrame = cv::imread("D:/linux_img_port/Fullscreen_2560x1440_three_five.png");

    myshop.updateFrame(missingSlotFrame);
    champNames = myshop.getChampNames();

    std::cout << "Champ Names:\n";
    for (const auto& name : champNames) {
        std::cout << name << " ";
    }
    std::cout << std::endl;

    std::cout << "Level: " << myshop.getLevel() << std::endl;
    std::cout << "Level Progress: " << myshop.getLevelProgress() << std::endl;
    std::cout << "Gold: " << myshop.getGold() << std::endl;

    delete myTime;
    
    
    fullScreen = cv::imread("D:/linux_img_port/Lvl/shop_one_three.png");

    myshop.relocateElements(ImageRegion(fullScreen, cv::Rect(0,0,fullScreen.cols,fullScreen.rows)));
    myshop.testElementIRVitals(fullScreen);


    */
    std::cin.get();

    return 0;
}
