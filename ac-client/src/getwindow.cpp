#include <Windows.h>
#include <overlayrect.cpp>
#include <opencv2/opencv.hpp>

// Capture the entire primary monitor as a cv::Mat
cv::Mat grabScreenMat() {
    HWND videoWnd = FindWindowA(NULL, "Media Player");
    if (!videoWnd) {
        MessageBoxA(NULL, "Can't find \'Media Player\' window!", "Error", MB_ICONERROR);
        return cv::Mat();
    }

    RECT vr; GetWindowRect(videoWnd, &vr);
    int vx = vr.left, vy = vr.top;
    int vw = vr.right - vr.left, vh = vr.bottom - vr.top;

    ShowWindow(videoWnd, SW_SHOWMAXIMIZED);
    SetForegroundWindow(videoWnd);
    SetFocus(videoWnd);
    UpdateWindow(videoWnd);

    Sleep(50);

    return getFrame(videoWnd);
}