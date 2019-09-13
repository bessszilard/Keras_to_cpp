################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../src/keras_model.cc 

CPP_SRCS += \
../src/keras_to_cpp.cpp 

CC_DEPS += \
./src/keras_model.d 

OBJS += \
./src/keras_model.o \
./src/keras_to_cpp.o 

CPP_DEPS += \
./src/keras_to_cpp.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"/media/szilard/3A7E80D57E808B79/_job2019/AI_motive/MNIST_toy_model/eclipse_proj/inc" -O0 -g3 -Wall -c -fmessage-length=0 -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"/media/szilard/3A7E80D57E808B79/_job2019/AI_motive/MNIST_toy_model/eclipse_proj/inc" -O0 -g3 -Wall -c -fmessage-length=0 -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


