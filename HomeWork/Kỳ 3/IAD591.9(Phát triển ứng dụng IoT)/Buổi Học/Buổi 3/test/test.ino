// C++ code
//

int bulb_in = 2;
int led_in = 4;
int piezo_in = 9;

void setup() {

  pinMode(bulb_in, OUTPUT);
  pinMode(led_in, OUTPUT);
  pinMode(piezo_in, OUTPUT);
}

void loop() {
  // piezo
  digitalWrite(piezo_in, HIGH);
  delay(3000);
  digitalWrite(piezo_in, LOW);
  delay(1000);

  digitalWrite(piezo_in, HIGH);
  delay(1000);
  digitalWrite(piezo_in, LOW);
  delay(1000);

  digitalWrite(piezo_in, HIGH);
  delay(1000);
  digitalWrite(piezo_in, LOW);
  delay(1000);

  // LED
  digitalWrite(led_in, HIGH);
  delay(1000);
  digitalWrite(led_in, LOW);
  delay(1000);

  // light bulb
  digitalWrite(bulb_in, HIGH);
  delay(2000);
  digitalWrite(bulb_in, LOW);
  delay(2000);
}