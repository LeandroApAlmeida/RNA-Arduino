// Este Sketch implementa uma Rede Neural Artificial usando o microcontrolador Arduino.
// A rede neural implementada é do modelo Multilayer Perceptron (MLP) com 10 neurônios
// na camada de entrada, duas camadas ocultas com 9 e 8 neurônios, respectivamente, e
// 3 neurônios na camada de saída.
//
// O exemplo não exige a construção de um circuito de demonstração, e o resultado do
// teste é exibido no console do IDE do Arduino.
//
// O funcionamento é o seguinte:
//
// 1. Assim que é implantado o programa, o LED da placa piscará algumas vezes e depois
// permancerá ligado. Isto indica que o arduino estará treinando a sua rede neural.
//
// 2. Após o treinamento, será impresso no console do IDE quais são as amostras com que
// a rede neural foi treinada, e quais as saídas esperadas para cada amostra.
//
// 3. O LED voltará a piscar novamente, logo após ele será desligado. Isto indica que
// o programa entrará no loop.
//
// 4. Durante o loop, o programa gerará um padrão aleatório, porém semelhante ao das
// amostras utilizadas para treinar a rede neural. Basicamente começa com uma sequência
// de 0's ou de 1's. Se começar com 0's, termina com 1's ou tem apenas 0's. Se começar 
// com 1's, termina com 0's ou tem apenas 1's. A cada 2 segundos um novo padrão é gerado,
// é feita a ativação/desativação dos pinos de entrada de acordo com este, e então são
// conectados estes pinos à camada de entrada da rede neural. É calculado as saídas esperadas
// para esta entrada. É conectada a camada de saída da rede neural aos pinos de saída
// do arduino 31, 33 e 35.
//
// Para ver o funcionamento na placa, recomendo que você conecte um multímetro nos pinos
// 31, 33 ou 35 no modo medidor de tensão contínua, e veja a alternância entre os valores 
// de 0 e 5V conforme as saídas são calculadas com base no padrão aleatório gerado. Como
// o padrão produz ou 0 ou 5 volts em todas estes três pinos de uma vez, então a escolha
// é indiferente de um ou de outro pino dentre estes três.
//
// Autor: Leandro Ap. Almeida


#define _1_OPTIMIZE B00010000
#define ACTIVATION__PER_LAYER
#define Sigmoid
#define Tanh

#define numberOf(arg) ((unsigned int) (sizeof (arg) / sizeof (arg [0]))) 

#include <NeuralNetwork.h>


// Pino que está conectado ao LED da placa.
const int OUTPUT_PIN_01 = 13; 
// Pinos conectados à camada de saída da rede neural (3 neurônios).
const int OUTPUT_PIN_02 = 31;
const int OUTPUT_PIN_03 = 33;
const int OUTPUT_PIN_04 = 35;


// Pinos conectados à camada de entrada da rede neural (10 neurônios).
const int INPUT_PIN_01 = 22;
const int INPUT_PIN_02 = 24;
const int INPUT_PIN_03 = 26;
const int INPUT_PIN_04 = 28;
const int INPUT_PIN_05 = 30;
const int INPUT_PIN_06 = 32;
const int INPUT_PIN_07 = 34;
const int INPUT_PIN_08 = 36;
const int INPUT_PIN_09 = 38;
const int INPUT_PIN_10 = 40;


// Conjunto de treinamento da rede neural.
const float trainingSet[14][10] = {
  //Começa com 0's e terminam com 1's,
  //ou tem somente 0's
  {0, 0, 0, 0, 1, 1, 1, 1, 1, 1},
  {0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
  {0, 0, 0, 0, 0, 0, 1, 1, 1, 1},
  {0, 0, 0, 0, 0, 0, 0, 1, 1, 1},
  {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  //Começa com 1's e terminam com 0's,
  //ou tem somente 1's.
  {1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
  {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
  {1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
  {1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
  {1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
};

// Saídas esperadas para o conjunto de treinamento
const float expectedOut[14][3] = {
  //Se começa com 0's e termina com 1's é {0, 0, 0}
  {0, 0, 0}, 
  {0, 0, 0}, 
  {0, 0, 0}, 
  {0, 0, 0},
  {0, 0, 0},
  {0, 0, 0},
  {0, 0, 0},
  //Se começa com 1's e termina com 0's é {1, 1, 1}
  {1, 1, 1}, 
  {1, 1, 1}, 
  {1, 1, 1},
  {1, 1, 1},
  {1, 1, 1},
  {1, 1, 1},
  {1, 1, 1}
};


// Estrutura da rede neural.
unsigned int layers[] = {
  10, // Camada de entrada: 10 neurônios.
  9,  // Camada oculta 1.
  8,  // Camada oculta 2.
  3   // Camada de saída: 3 neurônios.
};

// Função de ativação para cada camada oculta e a de saída.
byte activationFunctions[] = {
  1,
  1,
  0
};

// Cria a rede neural .
NeuralNetwork NN(layers, numberOf(layers), activationFunctions);


// Define o modo de cada um dos pinos que serão lidos
// ou escritos. Pinos de 22 a 40 (pares) estarão conectados
// à camada de entrada da rede neural. Pinos de 31 a 35
// (ímpares) estarão conectados à camada de saída da rede
// neural.
void setPinMode() {

  // AVISO MUITO IMPORTANTE!!!
  //   
  // ☝ ☝ ☝ ☝ ☝ ☝ ☝
  //
  // Os pinos definidos como OUTPUT trabalham todos no chamado
  // estado de baixa impedância, portanto, não tem um resistor
  // interno conectado a eles. Tome cuidado ao conectar qualquer
  // componente eletrônico como um LED, por exemplo, sem o uso
  // de um resistor adequado, que ele pode inutilizar o pino,
  // e no pior dos casos, até mesmo danificar o controlador
  // Atmega. Na dúvida, a melhor forma de testar os pinos é usando
  // um multímetro no modo de leitura de tensão.
  //
  // A corrente fornecida em cada pino fica em torno de 40 mA e tensão
  // de 5V.
  //
  // E de forma alguma conecte diretamente à placa componentes
  // como motores e relés, pois estes exigem uma corrente muito
  // maior que a fornecida nos pinos do Arduino, o que certamente
  // danificará permanentemente os mesmos ou até a placa inteira.

  pinMode(OUTPUT_PIN_01, OUTPUT);
  pinMode(OUTPUT_PIN_02, OUTPUT);
  pinMode(OUTPUT_PIN_03, OUTPUT);
  pinMode(OUTPUT_PIN_04, OUTPUT);

  pinMode(INPUT_PIN_01, OUTPUT);
  pinMode(INPUT_PIN_02, OUTPUT);
  pinMode(INPUT_PIN_03, OUTPUT);
  pinMode(INPUT_PIN_04, OUTPUT);
  pinMode(INPUT_PIN_05, OUTPUT);
  pinMode(INPUT_PIN_06, OUTPUT);
  pinMode(INPUT_PIN_07, OUTPUT);
  pinMode(INPUT_PIN_08, OUTPUT);
  pinMode(INPUT_PIN_09, OUTPUT);
  pinMode(INPUT_PIN_10, OUTPUT);

  digitalWrite(OUTPUT_PIN_01, LOW);
  digitalWrite(OUTPUT_PIN_02, LOW);
  digitalWrite(OUTPUT_PIN_03, LOW);
  digitalWrite(OUTPUT_PIN_04, LOW);

  digitalWrite(INPUT_PIN_01, LOW);
  digitalWrite(INPUT_PIN_02, LOW);
  digitalWrite(INPUT_PIN_03, LOW);
  digitalWrite(INPUT_PIN_04, LOW);
  digitalWrite(INPUT_PIN_05, LOW);
  digitalWrite(INPUT_PIN_06, LOW);
  digitalWrite(INPUT_PIN_07, LOW);
  digitalWrite(INPUT_PIN_08, LOW);
  digitalWrite(INPUT_PIN_09, LOW);
  digitalWrite(INPUT_PIN_10, LOW);

}


// Ativa o LED da placa para indicar que a rede neural está sendo
// treinada. Pisca o LED 3 vezes antes de ativá-lo.
void turnOnTheLED() {
  digitalWrite(OUTPUT_PIN_01, HIGH);
  delay(500);
  digitalWrite(OUTPUT_PIN_01, LOW);
  delay(500);
  digitalWrite(OUTPUT_PIN_01, HIGH);
  delay(500);
  digitalWrite(OUTPUT_PIN_01, LOW);
  delay(500);
  digitalWrite(OUTPUT_PIN_01, HIGH);
  delay(500);
  digitalWrite(OUTPUT_PIN_01, LOW);
  delay(500);
  digitalWrite(OUTPUT_PIN_01, HIGH);
}


// Desativa o LED da placa para indicar que a rede neural já foi
// treinada. Pisca o LED 3 vezes antes de desativá-lo.
void turnOffTheLED() {
  digitalWrite(OUTPUT_PIN_01, LOW);
  delay(500);
  digitalWrite(OUTPUT_PIN_01, HIGH);
  delay(500);
  digitalWrite(OUTPUT_PIN_01, LOW);
  delay(500);
  digitalWrite(OUTPUT_PIN_01, HIGH);
  delay(500);
  digitalWrite(OUTPUT_PIN_01, LOW);
  delay(500);
  digitalWrite(OUTPUT_PIN_01, HIGH);
  delay(500);
  digitalWrite(OUTPUT_PIN_01, LOW);
}


// Executa quando é implantado ou quando o botão reset na placa é pressionado.
void setup() {

  Serial.begin(9600);
  randomSeed(analogRead(0));
  setPinMode();
  turnOnTheLED();

  // Ajusta os pesos sinapticos com o treinamento da rede neural.
  do { 
    for (int i = 0; i < numberOf(trainingSet); i++) {
      NN.FeedForward(trainingSet[i]);
      NN.BackProp(expectedOut[i]);
    }
  } while(NN.getMeanSqrdError(numberOf(trainingSet)) > 0.003);

  // Imprime o conjunto de treinamento no console.
  Serial.print("\n\n");
  Serial.print("ENTRADAS:                      SAÍDAS ESPERADAS");

  for (int i = 0; i < 14; i++) {
    Serial.print("\n");
    for (int j= 0; j < 10; j++) {
      Serial.print((int) trainingSet[i][j]);
      Serial.print(" ");
    }
    Serial.print("    ->     ");
    for (int j= 0; j < 3; j++) {
      Serial.print((int) expectedOut[i][j]);
      Serial.print(" ");
    }
  }

  Serial.print("\n\n------------------------------------------------");
  Serial.print("\n\n");

  turnOffTheLED();

  Serial.print("ENTRADAS:                      SAÍDAS CALCULADAS\n");

}


// Faz a leitura dos pinos de entrada e calcula as saídas para os pinos lidos.
// Como não construí um circuito, o que faço aqui é simular a alternância nas
// entradas através de um processo de geração de números pseudoaleatórios.
void loop() {

  // Gera padrões de entrada semelhantes ao conjunto de treinamento, mas de modo
  // aleatório. Com isto, não há a geração de padrões muito diferentes do que a
  // rede neural foi treinada para reconhecer.
  // Os padrões de treinamento ou começam com 0's e terminam com 1's, ou começam
  // com 1's e terminam com 0's. Pode ser que os dez valores sejam somente 0's ou
  // somente 1's.
  int range = random(0, 11);
  int digit = random(0, 2);
  int reverse = digit == 1 ? 0 : 1;
  int pattern[10];
  for (int i = 0; i < range; i++) {
    pattern[i] = digit;
  }
  for (int i = range; i < 10; i++) {
    pattern[i] = reverse;
  }

  // Ativa/desativa os pinos de entrada de acordo com o padrão gerado.
  digitalWrite(INPUT_PIN_01, pattern[0]);
  digitalWrite(INPUT_PIN_02, pattern[1]);
  digitalWrite(INPUT_PIN_03, pattern[2]);
  digitalWrite(INPUT_PIN_04, pattern[3]);
  digitalWrite(INPUT_PIN_05, pattern[4]);
  digitalWrite(INPUT_PIN_06, pattern[5]);
  digitalWrite(INPUT_PIN_07, pattern[6]);
  digitalWrite(INPUT_PIN_08, pattern[7]);
  digitalWrite(INPUT_PIN_09, pattern[8]);
  digitalWrite(INPUT_PIN_10, pattern[9]);

  // Conecta os pinos de entrada à camada de entrada da rede neural.
  float input[10] = {
    digitalRead(INPUT_PIN_01),
    digitalRead(INPUT_PIN_02),
    digitalRead(INPUT_PIN_03),
    digitalRead(INPUT_PIN_04),
    digitalRead(INPUT_PIN_05),
    digitalRead(INPUT_PIN_06),
    digitalRead(INPUT_PIN_07),
    digitalRead(INPUT_PIN_08),
    digitalRead(INPUT_PIN_09),
    digitalRead(INPUT_PIN_10)
  };

  // Calcula as saídas de acordo com a leitura dos pinos de entrada.
  float *output = NN.FeedForward(input);

  // Conecta a camada de saída da rede neural com os pinos de
  // saída.
  digitalWrite(OUTPUT_PIN_02, round(output[0]));
  digitalWrite(OUTPUT_PIN_03, round(output[1]));
  digitalWrite(OUTPUT_PIN_04, round(output[2]));

  // Imprime no console do IDE o resultado do cálculo.
  for (int i = 0; i < numberOf(input); i++) {
    Serial.print(round(input[i]));
    Serial.print(" ");
  }
  Serial.print("    ->     ");
  Serial.print(digitalRead(OUTPUT_PIN_02));
  Serial.print(" ");
  Serial.print(digitalRead(OUTPUT_PIN_03));
  Serial.print(" ");
  Serial.print(digitalRead(OUTPUT_PIN_04));
  Serial.print("\n");

  free(output);

  delay(2000);

}