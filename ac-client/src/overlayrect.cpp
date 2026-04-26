#include <Windows.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <shop.h>


// Register barebones window class for our overlay
const char* OVERLAY_CLASS = "MyTransparentOverlay";

ATOM RegisterOverlayClass(HINSTANCE hInst) {
    WNDCLASSA wc = {};
    wc.lpfnWndProc = DefWindowProcA;       // no special messages
    wc.hInstance = hInst;
    wc.lpszClassName = OVERLAY_CLASS;
    return RegisterClassA(&wc);
}

// Create a borderless, layered, click-through window
HWND CreateOverlayWindow(HINSTANCE hInst, int x, int y, int w, int h) {
    HWND hwnd = CreateWindowExA(
        WS_EX_LAYERED    // enable layering
        | WS_EX_TRANSPARENT// click-through
        | WS_EX_TOPMOST,    // always on top
        OVERLAY_CLASS,      // window class
        "",                 // no title bar
        WS_POPUP,           // borderless
        x, y, w, h,
        NULL, NULL, hInst, NULL
    );
    if (hwnd) {
        // make magenta transparent
        SetLayeredWindowAttributes(hwnd, RGB(255, 0, 255), 0, LWA_COLORKEY);
        ShowWindow(hwnd, SW_SHOW);
    }
    return hwnd;
}

// convert cv::Rect to RECT (optionally offset by client to screen)
RECT CvToRECT(const cv::Rect& r, HWND target) {
    RECT out{ r.x, r.y, r.x + r.width, r.y + r.height };
    // map client to screen so drawing lines up with window
    POINT origin{ 0,0 };
    ClientToScreen(target, &origin);
    OffsetRect(&out, origin.x, origin.y);
    return out;
}


// Create mat based on current frame of the window
cv::Mat getFrame(HWND hwnd) {
    // Get client size
    RECT clientRC;
    GetClientRect(hwnd, &clientRC);
    int width = clientRC.right - clientRC.left;
    int height = clientRC.bottom - clientRC.top;

    // Convert client origin (0,0) to screen coords
    POINT topLeft = { clientRC.left, clientRC.top };
    ClientToScreen(hwnd, &topLeft);

    // Prepare device context and a bitmap
    HDC hdcScreen = GetDC(NULL);                     // whole screen
    HDC hdcMem = CreateCompatibleDC(hdcScreen);
    HBITMAP hBmp = CreateCompatibleBitmap(hdcScreen, width, height);
    SelectObject(hdcMem, hBmp);

    // Copy from screen DC at the client’s screen position
    BitBlt(
        hdcMem,              // target
        0, 0, width, height, // w/h in the mem DC
        hdcScreen,           // source: entire screen
        topLeft.x, topLeft.y,// clients positioning on the screen
        SRCCOPY
    );

    // transfer into a Mat (BGRA)
    BITMAPINFOHEADER bi = {};
    bi.biSize = sizeof(bi);
    bi.biWidth = width;
    bi.biHeight = -height;     // negative for topdown
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;

    cv::Mat src(height, width, CV_8UC4);
    GetDIBits(
        hdcMem, hBmp, 0, height,
        src.data, (BITMAPINFO*)&bi,
        DIB_RGB_COLORS
    );

    // Cleanup
    DeleteObject(hBmp);
    DeleteDC(hdcMem);
    ReleaseDC(NULL, hdcScreen);

    return src;
}