#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <shop.h>
#include <iostream>
#include <streambuf>
#include <ostream>
#include "getWindow.cpp"
#include "overlayrect.h"
#include "WINDOW.h"

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

struct DebugStreamBuf : std::streambuf {
    static const int BUF_SIZE = 512;
    char buffer[BUF_SIZE];

    DebugStreamBuf() {
        setp(buffer, buffer + BUF_SIZE - 1); // leave room for null-terminator
    }

    ~DebugStreamBuf() {
        sync();
    }

    int_type overflow(int_type ch) override {
        if (ch != traits_type::eof()) {
            *pptr() = ch;
            pbump(1);
        }
        return sync() == 0 ? ch : traits_type::eof();
    }

    int sync() override {
        if (pbase() == pptr()) return 0; // nothing to do
        int len = static_cast<int>(pptr() - pbase());
        buffer[len] = '\0';              // null-terminate
        ::OutputDebugStringA(buffer);
        pbump(-len);                     // reset put pointer
        return 0;
    }
};


int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int) {
    static DebugStreamBuf dbgBuf;
    std::ostream dbgOut(&dbgBuf);
    std::cout.rdbuf(&dbgBuf);
    std::cerr.rdbuf(&dbgBuf);

    // find the video window (replace with your player’s exact title!)
    HWND videoWnd = FindWindowA(NULL, WINDOW::NAME);
    if (!videoWnd) {
        MessageBoxA(NULL, "Can't find \'Media Player\' window!", "Error", MB_ICONERROR);
        return -1;
    }

    // Get its position & size
    RECT vr; GetWindowRect(videoWnd, &vr);
    int vx = vr.left, vy = vr.top;
    int vw = vr.right - vr.left, vh = vr.bottom - vr.top;

    // Create our overlay ON TOP of it
    if (!RegisterOverlayClass(hInst)) return -1;
    HWND overlay = CreateOverlayWindow(hInst, vx, vy, vw, vh);
    if (!overlay) return -1;

    MessageBox(nullptr, "Run screencap?", "Paused", MB_OK);

    ShowWindow(videoWnd, SW_SHOWMAXIMIZED);
    SetForegroundWindow(videoWnd);
    SetFocus(videoWnd);
    UpdateWindow(videoWnd);

    Sleep(50);

    //  get first frame
    cv::Mat firstFrame = getFrame(videoWnd);

    //  create Boxes
    cv::Mat shopTemplate = cv::imread("D:/linux_img_port/shop_2560x1440.PNG");
    Shop myShop;

    cv::Mat bgrOnly;
    cv::cvtColor(firstFrame, bgrOnly, cv::COLOR_BGRA2BGR);

    std::cout << "Locating shop and elements...\n";
    myShop.locateShopELements(ImageRegion(bgrOnly, cv::Rect(0, 0, bgrOnly.cols, bgrOnly.rows)), shopTemplate);
	myShop.testElementIRVitals(bgrOnly);
    std::vector<cv::Rect> boxes = myShop.getMajorBounds();
    std::string msg = "Vector size is: " + std::to_string(boxes.size());

    //  insert before the while(true) loop 
    std::vector<std::string> champNamesToDisplay;
    bool showChampNames = false;
	bool f2Pressed = false;

    while (true) {
        if (GetAsyncKeyState(VK_ESCAPE)) break;

        // capture the current frame
        cv::Mat currentFrame = getFrame(videoWnd); 

        // detect F2 key press to fetch and display champ names
        static bool f2Prev = false;
        bool f2Now = (GetAsyncKeyState(VK_F2) & 0x8000) != 0;
        if (f2Pressed && f2Now) {
            f2Pressed = false; // reset to avoid repeated triggers
			f2Now = false; // rest the current state
            champNamesToDisplay.clear(); // clear the names after displaying
        }
        else if (f2Now || f2Pressed/* && !f2Prev */ ) {
			f2Pressed = true;
			myShop.updateFrame(currentFrame);
            champNamesToDisplay = myShop.testAllString();
            showChampNames = true;
            // Show champ names in a message box for inspection
            std::string allNames;
            for (const auto& name : champNamesToDisplay) {
                allNames += name + "\n";
            }
            if (allNames.empty()) allNames = "(No names found)";
            //MessageBoxA(nullptr, allNames.c_str(), "Champ Names", MB_OK);
        }
        //f2Prev = f2Now;

        // Use currentFrame for further processing

        // Setup GDI offscreen DC & clear it
        HDC  hdcWin = GetDC(overlay);
        HDC  memDC = CreateCompatibleDC(hdcWin);
        HBITMAP bmp = CreateCompatibleBitmap(hdcWin, vw, vh);
        SelectObject(memDC, bmp);
        // Clear to magenta (transparent key)
        HBRUSH magenta = CreateSolidBrush(RGB(255, 0, 255));
        RECT full{ 0,0,vw,vh };
        FillRect(memDC, &full, magenta);
        DeleteObject(magenta);

		// draw fixed boxes
        HPEN pen = CreatePen(PS_SOLID, 3, RGB(0, 255, 0));
        HPEN oldP = (HPEN)SelectObject(memDC, pen);

        HBRUSH nullBrush = (HBRUSH)GetStockObject(NULL_BRUSH);
        HBRUSH oldBrush = (HBRUSH)SelectObject(memDC, nullBrush);
        for (auto& r : boxes) {
            RECT wr = CvToRECT(r, videoWnd);
            OffsetRect(&wr, -vx, -vy);
            Rectangle(memDC, wr.left, wr.top, wr.right, wr.bottom);
        }
        SelectObject(memDC, oldBrush);
        SelectObject(memDC, oldP);
        DeleteObject(pen);

        // draw champ names in top-left if flag is set
        if (showChampNames && !champNamesToDisplay.empty()) {
            int x = 20, y = 20;
            SetBkMode(memDC, TRANSPARENT);
            SetTextColor(memDC, RGB(255, 0, 0)); // Bright red
            HFONT hFont = CreateFont(24, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                                     ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
                                     DEFAULT_PITCH | FF_SWISS, "Arial");
            HFONT oldFont = (HFONT)SelectObject(memDC, hFont);
            for (const auto& name : champNamesToDisplay) {
                TextOutA(memDC, x, y, name.c_str(), (int)name.length());
                y += 28; // Line spacing
            }
            SelectObject(memDC, oldFont);
            DeleteObject(hFont);
        }

        // Bit-block trasnfer & cleanup
        BitBlt(hdcWin, 0, 0, vw, vh, memDC, 0, 0, SRCCOPY);
        DeleteObject(bmp);
        DeleteDC(memDC);
        ReleaseDC(overlay, hdcWin);

        // Pump messages so Windows doesn’t think app hung
        MSG msg;
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return 0;
}