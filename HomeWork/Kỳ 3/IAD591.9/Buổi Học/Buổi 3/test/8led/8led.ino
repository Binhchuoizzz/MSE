// ===== HC-SR04: TRIG D10, ECHO D9, 5V, GND =====
const uint8_t TRIG_PIN  = 10;
const uint8_t ECHO_PIN  = 9;
const unsigned long BAUD_RATE = 9600;

void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  Serial.begin(BAUD_RATE);
}

/* Gửi xung đo và trả về khoảng cách (cm)
   Trả về -1 nếu không nhận được echo trong thời gian chờ */
long readDistanceCM() {
  // Bước 1: đảm bảo TRIG ở LOW tối thiểu 2µs
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  // Bước 2: phát xung 10µs
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Bước 3: đo thời gian ECHO ở mức HIGH (us)
  // Thêm timeout 25ms ~ tầm xa ~ 4.3m để tránh kẹt
  unsigned long duration = pulseIn(ECHO_PIN, HIGH, 25000UL);

  if (duration == 0) return -1;              // ngoài tầm/không phản hồi
  long distance = duration / 58;             // ~ duration * 0.0343 / 2
  return distance;
}

void loop() {
  long d = readDistanceCM();

  if (d < 0) {
    Serial.println("Warning: no pulse from sensor");
  } else {
    Serial.print("Distance: ");
    Serial.print(d);
    Serial.println(" cm");
  }

  // Datasheet khuyên >=60ms giữa các lần đo để tránh nhiễu
  delay(60);
}
