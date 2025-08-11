// Gán chân theo thứ tự TRÁI → PHẢI
int led8 = 2;
int led7 = 3;
int led6 = 4;
int led5 = 5;
int led4 = 6;
int led3 = 7;
int led2 = 8;
int led1 = 9;

int leds[] = {led1, led2, led3, led4, led5, led6, led7, led8};

void setup() {
  for (int i = 0; i < 8; i++) pinMode(leds[i], OUTPUT);
}

// Bật N đèn đầu, tắt các đèn còn lại rồi giữ trong ms mili-giây
void firstN_on(int n, unsigned long ms) {
  for (int i = 0; i < 8; i++) digitalWrite(leds[i], (i < n) ? HIGH : LOW);
  delay(ms);
}
void allOn(unsigned long ms){ for(int i=0;i<8;i++) digitalWrite(leds[i],HIGH); delay(ms); }
void allOff(unsigned long ms){ for(int i=0;i<8;i++) digitalWrite(leds[i],LOW);  delay(ms); }

void loop() {
  // Bước 1: 4 đèn đầu sáng 2s, 4 đèn sau tắt
  firstN_on(4, 2000);

  // Bước 2: cả 8 đèn đều sáng 2s
  allOn(2000);

  // Bước 3: cả 8 đèn đều tắt 2s
  allOff(2000);

  // Bước 4: 2 đèn đầu sáng 2s, 6 đèn còn lại tắt
  firstN_on(2, 2000);

  // Bước 5: 4 đèn đầu sáng 2s, 4 đèn sau tắt
  firstN_on(4, 2000);

  // Bước 6: 6 đèn đầu sáng 2s, 2 đèn cuối tắt
  firstN_on(6, 2000);

  // Bước 7: 8 đèn đều sáng 2s
  firstN_on(8, 2000);
}
