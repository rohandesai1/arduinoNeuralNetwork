#include <math.h>
#include <stdlib.h>
#include <Arduino.h>

const int ledCount = 16;
int ledPins[ledCount] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, A0, A1, A2, A3}; // LED pins
float weights[ledCount];
unsigned long previousMillis[ledCount]; // Stores the last time LED was updated
bool ledStates[ledCount]; // Current state of each LED, true means ON
int flashCount = 0;


float xMatrix[10][3] = { 
    {19.81, 22.15, 130},
    {13.54, 14.36, 87.46},
    {13.08, 15.71, 85.63},
    {9.504, 12.44, 60.34},
    {15.34, 14.26, 102.5},
    {21.16, 23.04, 137.2},
    {16.65, 21.38, 110},
    {17.14, 16.4, 116},
    {13.03, 18.42, 82.61},
    {8.196, 16.84, 51.71}
};

float yMatrix[10][1] = {1, 0, 0, 0, 1, 1, 1, 1, 0, 0};

int iterations = 10; 

//Row 1: 2
//Row 2: 3,4,5,6
//Row 3: 7,8,9,10,11,12
//Row 4: 13, A0, A1, A2
//Row 5: A3

// Define the structure for a Value 
typedef struct {
    float data;
    float grad;
} Value;

// Define the structure for a Neuron
typedef struct {
    Value* weights;
    Value bias;
    float output; // to store the forward pass result
    int num_inputs;
} Neuron;

// Define the structure for a Layer
typedef struct {
    Neuron* neurons;
    int num_neurons;
} Layer;

// Define the structure for the MLP
typedef struct {
    Layer* layers;
    int num_layers;
} MLP;

// Random float between -1 and 1
float random_float() {
    return (float)rand() / (float)(RAND_MAX / 2) - 1;
}

// Sigmoid activation function
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Initialize a neuron with a given number of inputs
void init_neuron(Neuron* neuron, int num_inputs) {
    neuron->weights = (Value*)malloc(num_inputs * sizeof(Value));
    neuron->num_inputs = num_inputs;
    for (int i = 0; i < num_inputs; i++) {
        neuron->weights[i].data = random_float();
        neuron->weights[i].grad = 0.0;
    }
    neuron->bias.data = random_float();
    neuron->bias.grad = 0.0;
    neuron->output = 0.0; 
}

// Initialize a layer with a given number of neurons, each with a given number of inputs
void init_layer(Layer* layer, int num_neurons, int num_inputs) {
    layer->neurons = (Neuron*)malloc(num_neurons * sizeof(Neuron));
    layer->num_neurons = num_neurons;
    for (int i = 0; i < num_neurons; i++) {
        init_neuron(&layer->neurons[i], num_inputs);
    }
}

// Initialize an MLP with a given architecture
void init_mlp(MLP* mlp, int* sizes, int num_layers) {
    mlp->layers = (Layer*)malloc(num_layers * sizeof(Layer));
    mlp->num_layers = num_layers;
    for (int i = 0; i < num_layers; i++) {
        init_layer(&mlp->layers[i], sizes[i + 1], sizes[i]);
    }
}

// Forward pass for a single neuron
float forward_neuron(Neuron* neuron, float* inputs) {
    float sum = 0.0;
    for (int i = 0; i < neuron->num_inputs; i++) {
        sum += neuron->weights[i].data * inputs[i];
    }
    sum += neuron->bias.data;
    neuron->output = sigmoid(sum); // Store the output
    return neuron->output;
}

// Forward pass for a single layer
void forward_layer(Layer* layer, float* inputs, float* outputs) {
    for (int i = 0; i < layer->num_neurons; i++) {
        outputs[i] = forward_neuron(&layer->neurons[i], inputs);
    }
}

// Forward pass for the MLP
void forward_mlp(MLP* mlp, float* inputs, float* outputs) {
    float* current_inputs = inputs;
    float* current_outputs = (float*)malloc(mlp->layers[0].num_neurons * sizeof(float));

    for (int i = 0; i < mlp->num_layers; i++) {
        forward_layer(&mlp->layers[i], current_inputs, current_outputs);
        current_inputs = current_outputs;
        if (i < mlp->num_layers - 1) {
            current_outputs = (float*)malloc(mlp->layers[i + 1].num_neurons * sizeof(float));
        }
    }

    for (int i = 0; i < mlp->layers[mlp->num_layers - 1].num_neurons; i++) {
        outputs[i] = current_outputs[i];
    }

    free(current_outputs);
}

// Backward pass for a single neuron
void backward_neuron(Neuron* neuron, float* inputs, float grad_output) {
    float grad_sigmoid = grad_output * neuron->output * (1.0 - neuron->output);
    for (int i = 0; i < neuron->num_inputs; i++) {
        neuron->weights[i].grad += grad_sigmoid * inputs[i];
    }
    neuron->bias.grad += grad_sigmoid;
}

// Backward pass for a single layer
void backward_layer(Layer* layer, float* inputs, float* grad_outputs) {
    for (int i = 0; i < layer->num_neurons; i++) {
        backward_neuron(&layer->neurons[i], inputs, grad_outputs[i]);
    }
}

// Backward pass for the MLP
void backward_mlp(MLP* mlp, float* inputs, float* targets) {
    float* current_inputs = inputs;
    float* current_outputs = (float*)malloc(mlp->layers[0].num_neurons * sizeof(float));

    forward_mlp(mlp, current_inputs, current_outputs);
    float* grad_outputs = (float*)malloc(mlp->layers[mlp->num_layers - 1].num_neurons * sizeof(float));

    for (int i = 0; i < mlp->layers[mlp->num_layers - 1].num_neurons; i++) {
        grad_outputs[i] = current_outputs[i] - targets[i];
    }

    for (int i = mlp->num_layers - 1; i >= 0; i--) {
        backward_layer(&mlp->layers[i], current_inputs, grad_outputs);
        if (i > 0) {
            current_inputs = (float*)malloc(mlp->layers[i - 1].num_neurons * sizeof(float));
            forward_layer(&mlp->layers[i - 1], inputs, current_inputs);
        }
    }

    free(current_outputs);
    free(grad_outputs);
}

// Update weights and biases using the accumulated gradients
void update_parameters(MLP* mlp, float learning_rate) {
    for (int i = 0; i < mlp->num_layers; i++) {
        for (int j = 0; j < mlp->layers[i].num_neurons; j++) {
            for (int k = 0; k < mlp->layers[i].neurons[j].num_inputs; k++) {
                mlp->layers[i].neurons[j].weights[k].data -= learning_rate * mlp->layers[i].neurons[j].weights[k].grad;
                mlp->layers[i].neurons[j].weights[k].grad = 0.0;
            }
            mlp->layers[i].neurons[j].bias.data -= learning_rate * mlp->layers[i].neurons[j].bias.grad;
            mlp->layers[i].neurons[j].bias.grad = 0.0;
        }
    }
}

MLP mlp; // Declare the MLP globally 

int sizes[] = {1, 2, 3, 2, 1};
int num_layers = 3;

void setup() {
    Serial.begin(9600);
    // Initialize random seed
    randomSeed(analogRead(0));
    
    init_mlp(&mlp, sizes, num_layers);
    for (int i = 0; i < ledCount; i++) {
        pinMode(ledPins[i], OUTPUT); // Set each pin as an output
        previousMillis[i] = 0; // Initialize previousMillis
        ledStates[i] = false; // Start all LEDs off
      }

   
}

void loop() {

    unsigned long currentMillis = millis();

    for (int i = 0; i < ledCount; i++) {

      if (weights[i] <= -1) {
        digitalWrite(ledPins[i], LOW); // Ensure LED is off if intensity is -1
        continue; 
      }

      unsigned long interval = mapIntensityToIntervals(weights[i], ledStates[i]);
      
      if (currentMillis - previousMillis[i] >= interval) {
        flashCount += 1;
        // Change the state of the LED
        ledStates[i] = !ledStates[i];
        digitalWrite(ledPins[i], ledStates[i] ? HIGH : LOW);
        
        // Save the last time you changed the LED state
        previousMillis[i] = currentMillis;
      }

      if (flashCount > 5000){
        flashCount = 0;
        Serial.print(iterations);
        if (iterations >= 0){
          forwardBackFull();
        }
        iterations = iterations - 1;
      }
    }

}


void forwardBackFull() {
    // Input and target setup

    float* inputs = xMatrix[iterations];  
    float targets[1];  
    targets[0] = yMatrix[iterations][0];  


    float outputs[1]; 
    forward_mlp(&mlp, inputs, outputs);
    backward_mlp(&mlp, inputs, targets);
    update_parameters(&mlp, 1);

   


   /* for (int i = 0; i < num_layers; i++) {
        for (int j = 0; j < sizes[i + 1]; j++) {
            for (int k = 0; k < sizes[i]; k++) {
                Serial.print(mlp.layers[i].neurons[j].weights[k].data);
                Serial.print(", ");
            }
            Serial.print(mlp.layers[i].neurons[j].bias.data);
            Serial.print(", ");
        }
    }
    Serial.println();*/

    flashThroughAnimation(outputs);

}


void flashThroughAnimation(float* out) {

    for (int i = 0; i < ledCount; i++){
      digitalWrite(ledPins[i], LOW);
    }
 
    // Front prop
    for (int led = 0; led < ledCount; led++){
      unsigned long currentMillis = millis();
      while (millis() - currentMillis < 100){
        digitalWrite(ledPins[led], HIGH);
        delay(1);
        digitalWrite(ledPins[led], LOW);
        delay(mapIntensityToIntervals(weights[led], false));
      }
    }

    for (int flashes = 0; flashes < 3; flashes++){
      unsigned long currentMillis = millis();
      while (millis() - currentMillis < 1000){
        digitalWrite(ledPins[ledCount - 1], HIGH);
        delay(1);
        digitalWrite(ledPins[ledCount - 1], LOW);
        delay(mapIntensityToIntervals(out[0], false));
      }
      delay(500);
    }



    // Back prop
    for (int led = ledCount - 1; led >= 0; led--){
      unsigned long currentMillis = millis();
      while (millis() - currentMillis < 100){
        digitalWrite(ledPins[led], HIGH);
        delay(1);
        digitalWrite(ledPins[led], LOW);
        delay(mapIntensityToIntervals(weights[led], false));

      }
    }

  free(out);
    
}






unsigned long mapIntensityToIntervals(float intensity, bool ledState) {
  // Convert intensity from -1.0 to 1.0 to a delay interval
  if (ledState) {
    // Shorter on time, mainly constant to keep brightness control smoother
    return 1; // Very short on time for sharper contrast
  } else {
    return (1.0 - intensity) * 10; 
  }
}