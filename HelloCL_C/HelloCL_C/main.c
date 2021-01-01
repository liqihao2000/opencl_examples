//
//  main.c
//  HelloCL_C
//
//  Created by QI LI on 2021/1/1.
//  Copyright © 2021 QI LI. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>

int main (int argc, const char * argv[])
{
    cl_context context;
    cl_device_id *devices;
    char *devname;
    size_t cb;

    // create a GPU context
    context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
    if(context == 0) {
        printf("Can't create GPU context\n");
        return 0;
    }

    // get a list of devices
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    devices = (cl_device_id*) malloc(cb);
    clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, devices, 0);

    // show the name of the first device
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
    devname = (char*) malloc(cb);
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, devname, 0);
    printf("Device: %s\n", devname);

    // release everything
    free(devname);
    free(devices);
    clReleaseContext(context);

    return 0;
}
