#define NumberOf(arg) ((unsigned int) (sizeof (arg) / sizeof (arg [0]))) 

#define _1_OPTIMIZE B00010000
#define ACTIVATION__PER_LAYER
#define Sigmoid
#define Tanh

#include <NeuralNetwork.h>

// Conjunto de treinamento da rede neural.
const float train[8][10] = {
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
  {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
  {0, 0, 0, 0, 0, 0, 0, 1, 1, 1},
  {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
  {1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
};

// Saídas esperadas para o conjunto de treinamento
const float expected[8][3] = {
  {0, 0, 0}, 
  {0, 0, 0}, 
  {0, 0, 0}, 
  {0, 0, 0}, 
  {1, 1, 1}, 
  {1, 1, 1}, 
  {1, 1, 1}, 
  {1, 1, 1}
};

// Conjunto de teste da rede neural.
const float test[15][10] = {

  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
  {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
  {0, 0, 0, 0, 0, 0, 0, 1, 1, 1},
  {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
  {1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},

  {1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
  {0, 0, 0, 0, 0, 1, 1, 0, 0, 0},
  {0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
  {1, 1, 1, 1, 1, 1, 1, 1, 0, 1},
  {0, 0, 0, 0, 0, 0, 1, 1, 1, 0},
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {1, 1, 0, 0, 1, 0, 0, 0, 0, 1}
  
};

// Estrutura da rede neural.
unsigned int layers[] = {
  10, // Camada de entrada: 10 neurônios.
  9,  // Camada oculta 1.
  8,  // Camada oculta 2.
  3   // Camada de saída: 3 neurônios.
};

// Função de ativação para cada camada oculta e de saída.
byte Actv_Functions[] = {
  1,
  1,
  0
};

// Cria a rede neural .
NeuralNetwork NN(layers, NumberOf(layers), Actv_Functions);

// Saída calculada pela rede neural.
float *output;


// Executa quando o sketch é compilado ou quando o botão reset na placa é pressionado.
void setup() {

  Serial.begin(9600);

  // Imprime o conjunto de treinamento no console.
  Serial.print("\n\nCONJUNTO DE TREINAMENTO:\n\n");
  Serial.print("ENTRADAS:                      SAÍDAS ESPERADAS");
  for (int i = 0; i < 8; i++) {
    Serial.print("\n");
    for (int j= 0; j < 10; j++) {
      Serial.print((int) train[i][j]);
      Serial.print(" ");
    }
    Serial.print("    ->     ");
    for (int j= 0; j < 3; j++) {
      Serial.print((int) expected[i][j]);
      Serial.print(" ");
    }
  }

  // Imprime o conjunto de teste no console.
  Serial.print("\n\n\nCONJUNTO DE TESTE:\n");
  for (int i = 0; i < 15; i++) {
    Serial.print("\n");
    for (int j= 0; j < 10; j++) {
      Serial.print((int) test[i][j]);
      Serial.print(" ");
    }
  }

  // Ajusta os pesos sinapticos com o treinamento da rede neural.
  Serial.print("\n\n\nTREINANDO A REDE NEURAL...");
  do { 
    for (int i = 0; i < NumberOf(train); i++) {
      NN.FeedForward(train[i]);
      NN.BackProp(expected[i]);
    }
  } while(NN.getMeanSqrdError(NumberOf(train)) > 0.003);

  // Imprime as saídas calculadas para cada item de teste. Uso a função round
  // para ajustar para zero ou um, para se aproximar das saídas esperadas.
  Serial.println("\n\n\nSAÍDAS CALCULADAS PARA O CONJUNTO DE TESTE:\n");
  for (int i=0; i < NumberOf(test); i++) {
    output = NN.FeedForward(test[i]);
    Serial.print("[");
    Serial.print(output[0], 4);
    Serial.print(" | ");
    Serial.print(output[1], 4);
    Serial.print(" | ");
    Serial.print(output[2], 4);
    Serial.print("]");
    Serial.print(" ≅ ");
    Serial.print("[");
    Serial.print(round(output[0]));
    Serial.print(" | ");
    Serial.print(round(output[1]));
    Serial.print(" | ");
    Serial.print(round(output[2]));
    Serial.print("]");
    Serial.print("\n");
  }

}


// Controla a leitura das portas.
void loop() {

  // A ideia era obter as entradas a partir da leitura das portas 
  // digitais de 22 até 32. Os resultados seriam exibidos com
  // LED's conectador nas portas 40 a 42. Ligado seria 1, desligado, 0.

}