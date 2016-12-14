//standard includes
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <chrono>
#include <thread>

#include <vector>

//opencv includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//ZED Includes
#include <zed/Camera.hpp>


// Include OpenCV
#include <opencv2/opencv.hpp>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

#define GLM_FORCE_RADIANS
// Include GLM
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

using namespace glm;
using namespace cv;

#include "shader.hpp"
#include "texture.hpp"
#include "controls.hpp"
#include "objloader.hpp"
#include "orb_slam.h"
#include "planar_tracking.h"

using namespace sl::zed;
using namespace std;

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include "ORB_SLAM/System.h"

const int FPS = 30;
const ZEDResolution_mode ZED_RES = VGA;

Camera* zed;
cv::Mat Left;
//cv::Mat Depth;
cv::Mat Right;
int width, height;

bool stop_signal;

void grab_run()
{
    while (!stop_signal)
    {
        bool res = zed->grab(SENSING_MODE::STANDARD, 1, 1);

        if (!res)
        {
            slMat2cvMat(zed->retrieveImage(SIDE::LEFT)).copyTo(Left);
            //slMat2cvMat(zed->normalizeMeasure(MEASURE::DEPTH)).copyTo(Depth);
            slMat2cvMat(zed->retrieveImage(SIDE::RIGHT)).copyTo(Right);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    delete zed;
}

int main( void )
{
    zed = new Camera(ZED_RES, FPS);

    InitParams parameters;
    parameters.mode = QUALITY;
    parameters.unit = MILLIMETER;
    parameters.verbose = 1;

    ERRCODE err = zed->init(parameters);

    width = zed->getImageSize().width;
    height = zed->getImageSize().height;
    Left = cv::Mat(height, width, CV_8UC4, 1);
    //Depth = cv::Mat(height, width, CV_8UC4, 1);
    Right = cv::Mat(height, width, CV_8UC4, 1);

    std::thread grab_thread(grab_run);

    // Initialise Tracking System
    bool success = initTracking("../Extrinsics.xml");
    cv::Mat K = getCameraMatrix();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM("../Vocabulary/ORBvoc.bin","../zed_ygx.yaml",ORB_SLAM2::System::STEREO,true);

    if (!success)
        return 0;

    // Initialise GLFW
    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    window = glfwCreateWindow( 672, 376, "ZED-SLAM-AR", NULL, NULL);
    if( window == NULL ){
        fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    printf("OpenGL version supported by this platform (%s): \n", glGetString(GL_VERSION));

    // Initialize GLEW
    glewExperimental = GL_TRUE; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // Dark blue background
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

    // Cull triangles which normal is not towards the camera
    glEnable(GL_CULL_FACE);

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // Create and compile our GLSL program from the shaders
    GLuint programID = LoadShaders( "../Shaders/TransformVertexShader.vertexshader", "../Shaders/TextureFragmentShader.fragmentshader" );

    // Get a handle for our "MVP" uniform
    GLint MatrixID = glGetUniformLocation(programID, "MVP");

    // Load the texture
    int width, height;
    GLuint Texture = png_texture_load("../SpongeBob/spongebob.png", &width, &height);
    GLuint Texture1;
    glGenTextures(1, &Texture1);


    // Get a handle for our "myTextureSampler" uniform
    GLuint TextureID  = glGetUniformLocation(programID, "myTextureSampler");

    // Read our .obj file
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> normals; // Won't be used at the moment.
    loadOBJ("../SpongeBob/spongebob.obj", vertices, uvs, normals);

    // Load it into a VBO

    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

    GLuint uvbuffer;
    glGenBuffers(1, &uvbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
    glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), &uvs[0], GL_STATIC_DRAW);


    static const GLfloat g_vertex_buffer_data[] = {
            1.0f, -1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
            -1.0f,  -1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f,
            -1.0f,  -1.0f, 0.0f,
    };

    static const GLfloat g_uv_buffer_data[] = {
            1.0f, 0.0f,
            1.0f, 1.0f,
            0.0f,  0.0f,
            1.0f, 1.0f,
            0.0f,  1.0f,
            0.0f,  0.0f,
    };

    GLuint colorbuffer;
    glGenBuffers(1, &colorbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_uv_buffer_data), g_uv_buffer_data, GL_STATIC_DRAW);

    GLuint cubebuffer;
    glGenBuffers(1, &cubebuffer);
    glBindBuffer(GL_ARRAY_BUFFER, cubebuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

    bool slamMode = 0;

    Ptr<ORB> orb = ORB::create();
    orb->setScoreType(cv::ORB::FAST_SCORE);
    orb->setMaxFeatures(1000);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming(2)");
    Tracker orb_tracker(orb, matcher, K);
    orb_tracker.setFirstFrame("../template.png");

    cv::Mat frame_left;
    //cv::Mat frame_depth;
    cv::Mat frame_right;

    do{
        cvtColor(Left, frame_left, CV_BGRA2BGR);
        cvtColor(Right, frame_right, CV_BGRA2BGR);

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use our shader
        glUseProgram(programID);
        

        // Bind our texture in Texture Unit 0
        glActiveTexture(GL_TEXTURE1);
        Texture1 = loadframe_opencv(frame_left, Texture1);
        glBindTexture(GL_TEXTURE_2D, Texture1);


        // Set our "myTextureSampler" sampler to user Texture Unit 0
        glUniform1i(TextureID, 1);
        glm::mat4 MVP = glm::mat4(1.0);
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

        glDisable(GL_DEPTH_TEST);

        // 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, cubebuffer);
        glVertexAttribPointer(
                0,                  // attribute
                3,                  // size
                GL_FLOAT,           // type
                GL_FALSE,           // normalized?
                0,                  // stride
                (void*)0            // array buffer offset
        );

        // 2nd attribute buffer : UVs
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
        glVertexAttribPointer(
                1,                                // attribute
                2,                                // size
                GL_FLOAT,                         // type
                GL_FALSE,                         // normalized?
                0,                                // stride
                (void*)0                          // array buffer offset
        );
        glDrawArrays(GL_TRIANGLES, 0, 6 );

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);

        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        {
            slamMode = true;
        }

        success = orb_tracker.process(frame_left, slamMode);

        if (!success){
            glfwPollEvents();
            if (glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
                glfwWindowShouldClose(window) == 0){
                glfwSwapBuffers(window);
                continue;
            } else
                break;
        }

        glm::mat4 ViewMatrix;
        if (!slamMode)
        {
            ViewMatrix = getViewMatrix(slamMode);
        }
        else
        {
            cv::Mat CameraPose = SLAM.TrackStereo(frame_left, frame_right, 1);
            if(!CameraPose.empty())
            {
                trackStereo(CameraPose);
                ViewMatrix = getViewMatrix(slamMode);
            }
        }

        // Compute the MVP matrix from keyboard and mouse input
        computeMatricesFromInputs();
        
        glm::mat4 ProjectionMatrix = getProjectionMatrix();
        
        glm::mat4 ModelMatrix = getModelMatrix();

        glm::mat4 initModelMatrix = orb_tracker.getInitModelMatrix();

        MVP = ProjectionMatrix * ViewMatrix * initModelMatrix * ModelMatrix;

        glEnable(GL_DEPTH_TEST);

        // Send our transformation to the currently bound shader,
        // in the "MVP" uniform
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

        // Bind our texture in Texture Unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, Texture);

        // Set our "myTextureSampler" sampler to user Texture Unit 0
        glUniform1i(TextureID, 0);

        // 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glVertexAttribPointer(
                0,                  // attribute
                3,                  // size
                GL_FLOAT,           // type
                GL_FALSE,           // normalized?
                0,                  // stride
                (void*)0            // array buffer offset
        );

        // 2nd attribute buffer : UVs
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
        glVertexAttribPointer(
                1,                                // attribute
                2,                                // size
                GL_FLOAT,                         // type
                GL_FALSE,                         // normalized?
                0,                                // stride
                (void*)0                          // array buffer offset
        );

        // Draw the triangle !
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)vertices.size() );

        // Swap buffers
        glfwSwapBuffers(window);

        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        {
            slamMode = false;
            SLAM.Reset();
        }

        glfwPollEvents();

    } // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0 );

    stop_signal = true;

    grab_thread.join();

    // Cleanup VBO and shader
    glDeleteBuffers(1, &vertexbuffer);
    glDeleteBuffers(1, &uvbuffer);
    glDeleteProgram(programID);
    glDeleteTextures(1, &TextureID);
    glDeleteVertexArrays(1, &VertexArrayID);

    // Close OpenGL window and terminate GLFW
    glfwTerminate();

    return 0;
}

