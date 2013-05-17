/**********************************************************************
Copyright ?012 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

?Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
?Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/



// SimpleGLSample.cpp : Defines the entry point for the console application

#include "SimpleGL.hpp"
#ifndef _WIN32
#include <GL/glx.h>
#endif

#ifdef _WIN32
static HWND   gHwnd;
HDC           gHdc;
HGLRC         gGlCtx;
BOOL quit = FALSE;
MSG msg;
#else 
GLXContext gGlCtx;
#define GLX_CONTEXT_MAJOR_VERSION_ARB           0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB           0x2092
typedef GLXContext (*GLXCREATECONTEXTATTRIBSARBPROC)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
Window 		win;
Display 	*displayName;
XEvent          xev;
#endif

float theta = 0.0f;

cl_float animate = 0.0f;            /**< Animation rate */
//GLuint vertexObj;                   /**< Vertex object */
GLuint texture;                     /**< Texture */
GLuint glProgram;                   /**< GL program object */

int mouseOldX;                      /**< mouse controls */
int mouseOldY;
int mouseButtons = 0;
float rotateX    = 0.0f;
float rotateY    = 0.0f;
float translateZ = -3.0f;

clock_t t1, t2;
int frameCount = 0;
int frameRefCount = 90;
double totalElapsedTime = 0.0;

#define		GLSL_4CPP				1// GLSL replace cpp
#define    MATRIX_SIZE_LINE    3//3

#define    MEGA_SIZE     (1<<20)  // Mega, or million
#define    JOINT_SIZE    100

float    PROBLEM_SCALE[] ={ 0.25f, 0.5f, 1, 2, 4, 8, 16, 32 }; // 问题规模档次，8档，250K至32M，2倍递增
int    PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[2] ;// 问题规模, 初始设为1M，即一百万
int iClass=2;

#define STRINGIFY(A) #A

const char * vertexShader = STRINGIFY(
	uniform vec4 matrixLine[100 * 3];	
uniform int		boneNumber;  
attribute vec4  blendIndices ;
attribute vec4 blendWeights;
void main()
{
	vec4 blendPos = vec4(0, 0, 0, 0);
	int i=0;
	for (;i<boneNumber;i++)
	{
		int idx = int(blendIndices[i]+0.5) * 3 ;
		mat4 worldMatrix;
		worldMatrix[0] = matrixLine[idx];
		worldMatrix[1] = matrixLine[idx + 1];
		worldMatrix[2] = matrixLine[idx + 2];
		worldMatrix[3] = vec4(0);

		blendPos += worldMatrix * gl_Vertex * blendWeights[i];
	}

	gl_Position    = gl_ModelViewProjectionMatrix * blendPos;
}
);

const char * pixelShader = STRINGIFY(
void main()
{
  gl_FragColor       = vec4(1.0, 0.0, 0.0, 1.0);
}
);

#ifdef _WIN32
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
        
    case WM_CREATE:
        return 0;
        
    case WM_CLOSE:
        PostQuitMessage( 0 );
        return 0;
        
    case WM_DESTROY:
        return 0;

	case WM_LBUTTONUP:  
		mouseButtons = 0;  
		return 0;  

	case WM_LBUTTONDOWN:  
		mouseOldX = LOWORD(lParam);  
		mouseOldY = HIWORD(lParam);  
		mouseButtons = 1;  
		return 0; 
		 
	case WM_MOUSEMOVE:  
	{
		int x = LOWORD(lParam);  
		int y = HIWORD(lParam);  
		if (mouseButtons)   
		{  
	      int dx = x - mouseOldX;  
		  int dy = y - mouseOldY;  
          rotateX += (dy * 0.2f);  
          rotateY += (dx * 0.2f);  
		}  
		mouseOldX = x;   
		mouseOldY = y;
	}  
		return 0;

    case WM_KEYDOWN:
        switch ( wParam )
        {
            
        case VK_ESCAPE:
            PostQuitMessage(0);
            return 0;
            
        }
        return 0;
    
    default:
        return DefWindowProc( hWnd, message, wParam, lParam );
            
    }
}
#endif


#ifdef _WIN32
int
SimpleGLSample::enableGLAndGetGLContext(HWND hWnd, HDC &hDC, HGLRC &hRC, cl_platform_id platform, cl_context &context, cl_device_id &interopDevice)
{
    cl_int status;
    BOOL ret = FALSE;
    DISPLAY_DEVICE dispDevice;
    DWORD deviceNum;
    int  pfmt;
    PIXELFORMATDESCRIPTOR  pfd; 
    pfd.nSize           = sizeof(PIXELFORMATDESCRIPTOR); 
    pfd.nVersion        = 1; 
    pfd.dwFlags         = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER ;
    pfd.iPixelType      = PFD_TYPE_RGBA; 
    pfd.cColorBits      = 24; 
    pfd.cRedBits        = 8; 
    pfd.cRedShift       = 0; 
    pfd.cGreenBits      = 8; 
    pfd.cGreenShift     = 0; 
    pfd.cBlueBits       = 8; 
    pfd.cBlueShift      = 0; 
    pfd.cAlphaBits      = 8;
    pfd.cAlphaShift     = 0; 
    pfd.cAccumBits      = 0; 
    pfd.cAccumRedBits   = 0; 
    pfd.cAccumGreenBits = 0; 
    pfd.cAccumBlueBits  = 0; 
    pfd.cAccumAlphaBits = 0; 
    pfd.cDepthBits      = 24; 
    pfd.cStencilBits    = 8; 
    pfd.cAuxBuffers     = 0; 
    pfd.iLayerType      = PFD_MAIN_PLANE; 
    pfd.bReserved       = 0; 
    pfd.dwLayerMask     = 0;
    pfd.dwVisibleMask   = 0; 
    pfd.dwDamageMask    = 0;

    ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

    dispDevice.cb = sizeof(DISPLAY_DEVICE);

    DWORD displayDevices = 0;
    DWORD connectedDisplays = 0;

    int xCoordinate = 0;
    int yCoordinate = 0;
    int xCoordinate1 = 0;

    for (deviceNum = 0; EnumDisplayDevices(NULL, deviceNum, &dispDevice , 0); deviceNum++) 
    {
        if (dispDevice.StateFlags & DISPLAY_DEVICE_MIRRORING_DRIVER) 
        {
                continue;
        }

        if(!(dispDevice.StateFlags & DISPLAY_DEVICE_ACTIVE))
        {
            std::cout <<"Display device " << deviceNum << " is not connected!!" << std::endl;
            continue;
        }

        DEVMODE deviceMode;

        // initialize the DEVMODE structure
        ZeroMemory(&deviceMode, sizeof(deviceMode));
        deviceMode.dmSize = sizeof(deviceMode);
        deviceMode.dmDriverExtra = 0;

        EnumDisplaySettings(dispDevice.DeviceName, ENUM_CURRENT_SETTINGS, &deviceMode);

        xCoordinate = deviceMode.dmPosition.x;
        yCoordinate = deviceMode.dmPosition.y;

        WNDCLASS windowclass;

        windowclass.style = CS_OWNDC;
        windowclass.lpfnWndProc = WndProc;
        windowclass.cbClsExtra = 0;
        windowclass.cbWndExtra = 0;
        windowclass.hInstance = NULL;
        windowclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
        windowclass.hCursor = LoadCursor(NULL, IDC_ARROW);
        windowclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
        windowclass.lpszMenuName = NULL;
        windowclass.lpszClassName = reinterpret_cast<LPCSTR>("SimpleGL");
        RegisterClass(&windowclass);

        gHwnd = CreateWindow(reinterpret_cast<LPCSTR>("SimpleGL"), 
                              reinterpret_cast<LPCSTR>("OpenGL Texture Renderer"), 
                              WS_CAPTION | WS_POPUPWINDOW, 
                              isDeviceIdEnabled() ? xCoordinate1 : xCoordinate, 
                              yCoordinate, 
                              screenWidth, 
                              screenHeight, 
                              NULL, 
                              NULL, 
                              windowclass.hInstance, 
                              NULL);
        hDC = GetDC(gHwnd);

        pfmt = ChoosePixelFormat(hDC, 
                    &pfd);
        if(pfmt == 0) 
        {
            std::cout << "Failed choosing the requested PixelFormat.\n";
            return SDK_FAILURE;
        }

        ret = SetPixelFormat(hDC, pfmt, &pfd);
        if(ret == FALSE) 
        {
            std::cout << "Failed to set the requested PixelFormat.\n";
            return SDK_FAILURE;
        }
        
        hRC = wglCreateContext(hDC);
        if(hRC == NULL) 
        {
            std::cout << "Failed to create a GL context"<<std::endl;
            return SDK_FAILURE;
        }

        ret = wglMakeCurrent(hDC, hRC);
        if(ret == FALSE) 
        {
            std::cout << "Failed to bind GL rendering context";
            return SDK_FAILURE;
        }	
        displayDevices++;

        cl_context_properties properties[] = 
        {
                CL_CONTEXT_PLATFORM, (cl_context_properties) platform,
                CL_GL_CONTEXT_KHR,   (cl_context_properties) hRC,
                CL_WGL_HDC_KHR,      (cl_context_properties) hDC,
                0
        };
        
        if (!clGetGLContextInfoKHR) 
        {
               clGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn) clGetExtensionFunctionAddressForPlatform(platform, "clGetGLContextInfoKHR");
               if (!clGetGLContextInfoKHR) 
               {
                    std::cout << "Failed to query proc address for clGetGLContextInfoKHR";
                    return SDK_FAILURE;
               }
        }
        
        size_t deviceSize = 0;
        status = clGetGLContextInfoKHR(properties, 
                                      CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
                                      0, 
                                      NULL, 
                                      &deviceSize);
        CHECK_OPENCL_ERROR(status, "clGetGLContextInfoKHR failed!!");

        if (deviceSize == 0) 
        {
            // no interopable CL device found, cleanup
            wglMakeCurrent(NULL, NULL);
            wglDeleteContext(hRC);
            DeleteDC(hDC);
            hDC = NULL;
            hRC = NULL;
            DestroyWindow(gHwnd);
            // try the next display
            continue;
        }
        else 
        {
            if (deviceId == 0)
            {
                ShowWindow(gHwnd, SW_SHOW);
                //Found a winner 
                break;
            }
            else if (deviceId != connectedDisplays)
            {
                connectedDisplays++;
                wglMakeCurrent(NULL, NULL);
                wglDeleteContext(hRC);
                DeleteDC(hDC);
                hDC = NULL;
                hRC = NULL;
                DestroyWindow(gHwnd);
                if (xCoordinate >= 0)
                {
                    xCoordinate1 += deviceMode.dmPelsWidth;
                    // try the next display
                }
                else 
                {
                    xCoordinate1 -= deviceMode.dmPelsWidth;
                }

                continue;
            } 
            else 
            {
                ShowWindow(gHwnd, SW_SHOW);
                //Found a winner 
                break;
            }
        }

    }

    if (!hRC || !hDC) 
    {
       OPENCL_EXPECTED_ERROR("OpenGL interoperability is not feasible.");
    }

    cl_context_properties properties[] = 
    {
        CL_CONTEXT_PLATFORM, (cl_context_properties) platform,
        CL_GL_CONTEXT_KHR,   (cl_context_properties) hRC,
        CL_WGL_HDC_KHR,      (cl_context_properties) hDC,
        0
    };


    if (deviceType.compare("gpu") == 0)
    {
        status = clGetGLContextInfoKHR( properties, 
                                        CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
                                        sizeof(cl_device_id), 
                                        &interopDevice, 
                                        NULL);
        CHECK_OPENCL_ERROR(status, "clGetGLContextInfoKHR failed!!");

        // Create OpenCL context from device's id
        context = clCreateContext(properties,
                                         1,
                                         &interopDevice,
                                         0,
                                         0,
                                         &status);
        CHECK_OPENCL_ERROR(status, "clCreateContext failed!!");
        std::cout<<"Interop Device Id "<<interopDevice<<std::endl;
    }
    else 
    {
        context = clCreateContextFromType(
                    properties,
                    CL_DEVICE_TYPE_CPU,
                    NULL,
                    NULL,
                    &status);
        CHECK_OPENCL_ERROR(status, "clCreateContextFromType failed!!");
    }

    // OpenGL animation code goes here

    // GL init
    glewInit();
    if (! glewIsSupported("GL_VERSION_2_0 " "GL_ARB_pixel_buffer_object"))
    {
          std::cerr << "Support for necessary OpenGL extensions missing."
                    << std::endl;
          return SDK_FAILURE;
    }

    glEnable(GL_TEXTURE_2D);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(
        60.0,
        (GLfloat)WINDOW_WIDTH / (GLfloat) WINDOW_HEIGHT,
        0.1,
        10.0);

    return SDK_SUCCESS;
}

void 
SimpleGLSample::disableGL(HWND hWnd, HDC hDC, HGLRC hRC)
{
    wglMakeCurrent( NULL, NULL );
    wglDeleteContext( hRC );
    ReleaseDC( hWnd, hDC );
}
#endif

int 
SimpleGLSample::setupSL()
{
	cl_int status = CL_SUCCESS;

    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */
    cl_platform_id platform = NULL;
    int retValue = sampleCommon->getPlatform(platform, platformId, isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "sampleCommon::getPlatform() failed");

    // Display available devices.
    //retValue = sampleCommon->displayDevices(platform, dType);
   // CHECK_ERROR(retValue, SDK_SUCCESS, "sampleCommon::displayDevices() failed");

    retValue = enableGLAndGetGLContext(gHwnd, gHdc, gGlCtx, platform, context, interopDeviceId);
    if (retValue != SDK_SUCCESS)
    {
        return retValue;
    }

    // Compile Vertex and Pixel shaders and create glProgram
	glProgram = compileProgram(vertexShader, pixelShader);
	if(!glProgram)
	{
        std::cout << "ERROR: Failed to create glProgram " << std::endl;
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

int 
SimpleGLSample::setup()
{
  
    cl_int retValue = setupSL();
    if (retValue != SDK_SUCCESS)
    return retValue;

	PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[iClass] ;
	mvm.initialize( PROBLEM_SIZE, JOINT_SIZE , sampleCommon, &_timeValueList );

	mvm.setupVBO( context, interopDeviceId, kernel, commandQueue , _locationAttrib);

    return SDK_SUCCESS;
}

void
SimpleGLSample::SimpleGLCPUReference(void)
{
    for(unsigned int i = 0; i < meshHeight; ++i)
    {
        for(unsigned int j = 0; j < meshWidth; ++j)
        {
            unsigned int x = j;
            unsigned int y = i;

            // calculate uv coordinates
            float u = x / (float)meshWidth;
            float v = y / (float)meshHeight;
            u = u * 2.0f - 1.0f;
            v = v * 2.0f - 1.0f;

            // calculate simple sine wave pattern
            float freq = 4.0f;
            float w = sin(u * freq + animate) * cos(v * freq + animate) * 0.5f;

            // write output vertex
            refPos[i * meshWidth * 4 + j * 4 + 0] = u;
            refPos[i * meshWidth * 4 + j * 4 + 1] = w;
            refPos[i * meshWidth * 4 + j * 4 + 2] = v;
            refPos[i * meshWidth * 4 + j * 4 + 3] = 1.0f;
        }
    }
}

int 
SimpleGLSample::initialize()
{
     // Call base class Initialize to get default configuration
    if (this->SDKSample::initialize() != SDK_SUCCESS)
            return SDK_FAILURE;
	
	for (int i=0;i< _initTimerCount;i++)
	{
		int timer = createTimer();
		_timers.push_back(timer);
	}

    return SDK_SUCCESS;
}

int 
SimpleGLSample::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
	mvm.unInitialize();

	return SDK_SUCCESS;
}

int 
SimpleGLSample::genBinaryImage()
{
    streamsdk::bifData binaryData;
    binaryData.kernelName = std::string("SimpleGL_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(isComplierFlagsSpecified())
        binaryData.flagsFileName = std::string(flags.c_str());

    binaryData.binaryName = std::string(dumpBinary.c_str());
    int status = sampleCommon->generateBinaryImage(binaryData);
    return status;
}

int 
SimpleGLSample::verifyResults()
{
	//Do verification
	printf("Performing verification...\n");
	bool result = mvm.verifyEqual( );
	printf("%s", !result ?"ERROR: Verification failed.\n":"Verification succeeded.\n");

     if(verify)
    {
        // it overwrites the input array with the output
        refPos = (cl_float*)malloc(meshWidth * meshHeight * sizeof(cl_float4));
        CHECK_ALLOCATION(refPos, "Failed to allocate host memory. (refPos)");

        memset(refPos, 0, meshWidth * meshHeight * sizeof(cl_float4));
        SimpleGLCPUReference();

        // compare the results and see if they match
        if(compareArray(pos, refPos, meshWidth * meshHeight * 4))
        {
            std::cout << "Failed\n" << std::endl;
            return SDK_FAILURE;
        }
        else
        {
            std::cout << "Passed!\n" << std::endl;
            return SDK_SUCCESS;
        }
    }
    return SDK_SUCCESS;
}

int 
SimpleGLSample::run()
{
    int status = 0;

    if(!quiet && !verify)
    {
#ifdef _WIN32
        // program main loop
        while (!quit)
        {
            // check for messages
            if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
            {
                // handle or dispatch messages
                if (msg.message == WM_QUIT) 
                {
                    quit = TRUE;
                } 
                else 
                {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
            } 
            else 
            {
				int timer1 = getTimerCurrent(1);
				resetTimer(timer1);
				startTimer(timer1);

                // OpenGL animation code goes here		

                t1 = clock() * CLOCKS_PER_SEC;
                frameCount++;

				// run CPP kernel to generate vertex positions
				int timer = getTimerCurrent(0);
				resetTimer(timer);
				startTimer(timer);

#if !GLSL_4CPP
#if 1//!VECTOR_FLOAT4
				mvm.ExecuteNativeCPP();
#else
				mvm.ExecuteNativeSSE();
#endif
#endif
				stopTimer(timer);
				double dTime = (cl_double)readTimer(timer);
				insertTimer("2.executeKernelCPP", dTime);

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                // set view matrix
                glMatrixMode(GL_MODELVIEW);
                glLoadIdentity();
                glTranslatef(0.0, 0.0, translateZ);
                glRotatef(rotateX, 1.0, 0.0, 0.0);
                glRotatef(rotateY, 0.0, 1.0, 0.0);
#if 1
				resetTimer(timer);
				startTimer(timer);

                // render from the vbo

				glUniform1i( _locationUniform[1], SIZE_PER_BONE );
				glUniform4fv( _locationUniform[0], mvm._joints.nSize * MATRIX_SIZE_LINE, (float*)mvm._joints.pMatrix );
				
#if GLSL_4CPP				
				glUseProgram(glProgram);
#endif

				mvm.renderVBO();

                glFinish();

				stopTimer(timer);
				dTime = (cl_double)readTimer(timer);
				insertTimer("2.render", dTime);
#endif
 
				SwapBuffers(gHdc);
               t2 = clock() * CLOCKS_PER_SEC;
                totalElapsedTime += (double)(t2 - t1);
                if(1&&frameCount && frameCount > frameRefCount)
                {

                    // set  Window Title
                    char title[256];
                    double fMs = (double)((totalElapsedTime / (double)CLOCKS_PER_SEC) / (double) frameCount);
                    int framesPerSec = (int)(1.0 / (fMs / CLOCKS_PER_SEC));
            #if defined (_WIN32) && !defined(__MINGW32__)
                    sprintf_s(title, 256, "OpenCL SimpleGL | %d fps : %lf s", framesPerSec, fMs/ CLOCKS_PER_SEC);
            #else 
                    sprintf(title, "OpenCL SimpleGL | %d fps ", framesPerSec);
            #endif
                    SetWindowText(gHwnd, title);
                    frameCount = 0;
                    totalElapsedTime = 0.0;
                }

                animate += 0.01f;

				stopTimer(timer1);
				double dTime1 = (cl_double)readTimer(timer1);
				insertTimer("run", dTime1);
            }
            
        }
#else 
    // OpenGL animation code goes here              
    XSelectInput(displayName, 
    win, 
    ExposureMask | KeyPressMask | ButtonPressMask | ButtonReleaseMask | Button1MotionMask);
    while(1)
    {
        /* handle the events in the queue */
        while (XPending(displayName) > 0)
        {
            XNextEvent(displayName, &xev);
            switch (xev.type)
            {
                /* exit in case of a mouse button press */
                case ButtonPress:
            if (xev.xbutton.button == Button2)//Exit when middle mouse button is pressed
                    {
                        glXMakeCurrent(displayName, None, NULL);
                        glXDestroyContext(displayName, gGlCtx);
                        XDestroyWindow(displayName, win);
                        XCloseDisplay(displayName);
                        exit(0);
                    }
                    else if (xev.xbutton.button == Button1)//When left mouse buttomn is pressed
                    {
            mouseButtons = 1 ;
                        //Get the x, y values of mouse pointer
                        mouseOldX = xev.xbutton.x;
                        mouseOldY = xev.xbutton.y;
                    }
                    break; 	
                
        case ButtonRelease:
                    if (xev.xbutton.button == Button1)//When left mouse button is released
                    {
            mouseButtons = 0;
            mouseOldX = xev.xbutton.x;
                        mouseOldY = xev.xbutton.y;
                    }
                    break;
        case MotionNotify:
             float dx, dy;
                 dx = xev.xbutton.x - mouseOldX;
                 dy = xev.xbutton.y - mouseOldY;

                 if (mouseButtons)
                 {
                rotateX += static_cast<float>(dy * 0.2);
                rotateY += static_cast<float>(dx * 0.2);
                 }

                 mouseOldX = xev.xbutton.x;
                 mouseOldY = xev.xbutton.y;
            break;
        case KeyPress:
                    char buf[2];
                    int len;
                    KeySym keysym_return;
                    len = XLookupString(&xev.xkey, buf, 1, &keysym_return, NULL);
                   
                    if ( len != 0 )
            {
            if(buf[0] == (char)(27))//Escape character
            {
                glXMakeCurrent(displayName, None, NULL);
                            glXDestroyContext(displayName, gGlCtx);
                            XDestroyWindow(displayName, win);
                            XCloseDisplay(displayName);
                            exit(0);			
            }
                    }
                    break;
            }
        }
        t1 = clock() * CLOCKS_PER_SEC;
        frameCount++;

        // run OpenCL kernel to generate vertex positions
        executeKernel();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // set view matrix
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0, 0.0, translateZ);
        glRotatef(rotateX, 1.0, 0.0, 0.0);
        glRotatef(rotateY, 0.0, 1.0, 0.0);

        // render from the vbo
        glBindBuffer(GL_ARRAY_BUFFER, vertexObj);
        glVertexPointer(4, GL_FLOAT, 0, 0);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);

        glUseProgram(glProgram);
        glEnableClientState(GL_VERTEX_ARRAY);
        glColor3f(1.0, 0.0, 0.0);
        glDrawArrays(GL_POINTS, 0, WINDOW_WIDTH * WINDOW_HEIGHT);
        glDisableClientState(GL_VERTEX_ARRAY);

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glFinish();
        glXSwapBuffers (displayName, win);
        t2 = clock() * CLOCKS_PER_SEC;
        totalElapsedTime += (double)(t2 - t1);
        if(frameCount && frameCount > frameRefCount)
        {
            // set Window Title
            char title[256];
            double fMs = (double)((totalElapsedTime / (double)CLOCKS_PER_SEC) / (double) frameCount);
            int framesPerSec = (int)(1.0 / (fMs / CLOCKS_PER_SEC));
        #if defined (_WIN32) && !defined(__MINGW32__)
            sprintf_s(title, 256, "OpenCL SimpleGL | %d fps ", framesPerSec);
        #else
            sprintf(title, "OpenCL SimpleGL | %d fps ", framesPerSec);
        #endif
            XStoreName(displayName, win, title);
            frameCount = 0;
            totalElapsedTime = 0.0;
         }

         animate += 0.01f;
    }
#endif
    }
   
    return SDK_SUCCESS;
}

GLuint SimpleGLSample::compileProgram(const char * vsrc, const char * psrc)
{
    GLint err = 0;

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint pixelShader  = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vsrc, 0);
    glShaderSource(pixelShader, 1, &psrc, 0);

    glCompileShader(vertexShader);

    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &err);

    if(!err)
    {
        char temp[256];
        glGetShaderInfoLog(vertexShader, 256, 0, temp);
        std::cout << "Failed to compile shader: " << temp << std::endl;
        return SDK_FAILURE;
    }

    glCompileShader(pixelShader); 

    glGetShaderiv(pixelShader, GL_COMPILE_STATUS, &err);

    if(!err)
    {
        char temp[256];
        glGetShaderInfoLog(pixelShader, 256, 0, temp);
        std::cout << "Failed to compile shader: " << temp << std::endl;
        return SDK_FAILURE;
    }

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, pixelShader);

    glLinkProgram(program);

	_locationUniform[0] = glGetUniformLocation( program, "matrixLine");
	_locationUniform[1] = glGetUniformLocation( program, "boneNumber");
	_locationAttrib[0] = glGetAttribLocation( program, "blendIndices");
	_locationAttrib[1] = glGetAttribLocation( program, "blendWeights");

	int maxUniformElment ;
	glGetIntegerv( GL_MAX_VERTEX_UNIFORM_COMPONENTS, &maxUniformElment );
	std::cout << "GL_MAX_VERTEX_UNIFORM_COMPONENTS = " << maxUniformElment << std::endl;

    // check if program linked
    err = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &err);

    if(!err)
    {
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        std::cout << "Failed to link program: " << temp << std::endl;
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}

int SimpleGLSample::loadTexture(GLuint * texture)
{
    std::string imagePath = sampleCommon->getPath();
    imagePath.append("SimpleGL.bmp");

    streamsdk::SDKBitMap image(imagePath.c_str());
    if (!image.isLoaded())
    {
        std::cout << "ERROR: could not load bitmap : " << imagePath.c_str() << std::endl;
        return SDK_FAILURE;
    }

    glGenTextures(1, texture );

    glBindTexture(GL_TEXTURE_2D, *texture);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA8,
        image.getWidth(),
        image.getHeight(),
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        image.getPixels());

    return SDK_SUCCESS;
}

int
SimpleGLSample::compareArray(const float* mat0,
                    const float* mat1,
                    unsigned int size)
{
    const float epsilon = (float)1e-2;
    for (unsigned int i = 0; i < size; ++i)
    {
        float val0 = mat0[i];
        float val1 = mat1[i];

        float diff = (val1 - val0);
        if (fabs(val1) > epsilon)
        {
            diff /= val0;
        }

        if(fabs(diff) > epsilon)
            return (fabs(diff) > epsilon);
    }

    return SDK_SUCCESS;
}

SimpleGLSample::~SimpleGLSample()
{
    // release program resources
    FREE(pos);
    FREE(refPos);
    FREE(devices);
}

void SimpleGLSample::printfTimer()
{
	for (TimerListItr itr=_timeValueList.begin(); itr!=_timeValueList.end(); itr++)
	{
		std::cout << itr->first << ":  " << itr->second << std::endl;
	}
	std::cout << std::endl;
}

void SimpleGLSample::insertTimer( std::string item, double time)
{
	if ( _timeValueList.size()>100 )
	{
		return;
	}
	_timeValueList.insert( std::make_pair(item, time) );
}

SimpleGLSample *SimpleGLSample::simpleGLSample = NULL;


int 
main(int argc, char* argv[])
{
    int status = 0;

    SimpleGLSample glSampleObj("Simple OpenGL Sample");	

    SimpleGLSample::simpleGLSample = &glSampleObj;

    if (glSampleObj.initialize() != SDK_SUCCESS)
        return SDK_FAILURE;

    if (glSampleObj.parseCommandLine(argc, argv) != SDK_SUCCESS)
        return SDK_FAILURE;

    if(glSampleObj.isDumpBinaryEnabled())
    {
        return glSampleObj.genBinaryImage();
    }

    status = glSampleObj.setup();
    if(status != SDK_SUCCESS)
        return (status == SDK_EXPECTED_FAILURE) ? SDK_SUCCESS : SDK_FAILURE;

	int timer = glSampleObj.getTimerCurrent();
	glSampleObj.resetTimer(timer);
	glSampleObj.startTimer(timer);

	if (glSampleObj.run() != SDK_SUCCESS)
		return SDK_FAILURE;

	glSampleObj.stopTimer(timer);
	double dTime = (cl_double)glSampleObj.readTimer(timer);
	glSampleObj.insertTimer("run", dTime);

#ifdef _WIN32
    glSampleObj.disableGL(gHwnd, gHdc, gGlCtx);
#endif

    if (glSampleObj.verifyResults() != SDK_SUCCESS)
        return SDK_FAILURE;

    if (glSampleObj.cleanup() != SDK_SUCCESS)
        return SDK_FAILURE;
	
	glSampleObj.printfTimer();

    return SDK_SUCCESS;
}

