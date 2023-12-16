#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>



/*
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>

// Datasheet: https://cdn-shop.adafruit.com/datasheets/BST_BNO055_DS000_12.pdf


class Adafruit_BNO055_ext : public Adafruit_BNO055 {   
  // Page 27
  typedef enum {
    G2 = 0b00,
    G4 = 0b01,
    G8 = 0b10,
    G16 = 0b11
  } adafruit_bno055_acc_range_t;

  typedef enum {
    Hz7_81 = 0b000,
    Hz15_63 = 0b001,
    Hz31_25 = 0b010,
    Hz62_5 = 0b011,
    Hz125 = 0b100,
    Hz250 = 0b101,
    Hz500 = 0b110,
    Hz1000 = 0b111
  } adafruit_bno055_acc_bw_t;


  void update_range(adafruit_bno055_acc_range_t value) {
    if (value > 0b11) {
      return -1;
    }
    write_config(BNO055_ACC_CONFIG_ADDR, value, 0b11);
  }

  void update_bandwidth(adafruit_bno055_acc_bw_t value) {
    update_bits(BNO055_ACC_CONFIG_ADDR, value << 2, 0b111 << 2);
  }


  void update_bits(adafruit_bno055_reg_t addr, byte value, byte mask) {
    byte old_value = read8(addr);
    byte new_value = (old_value & ~mask) | (value & mask);
    write_config(BNO055_ACC_CONFIG_ADDR, new_value);
  }

  void update_units(bool use_mg) {
    update_bits(BNO055_UNIT_SEL_ADDR, use_mg);
  }

  void write_config(adafruit_bno055_reg_t addr, byte value)
  {
    adafruit_bno055_opmode_t modeback = _mode;

    // Switch to config mode (just in case since this is the default)
    setMode(OPERATION_MODE_CONFIG);
    delay(25);
    // save selected page ID and switch to page 1
    uint8_t savePageID = read8(BNO055_PAGE_ID_ADDR); // Page 53 in datasheet
    write8(BNO055_PAGE_ID_ADDR, 0x01);

    // set configuration to 2G range
    write8(addr, value);
    delay(10);

    // restore page ID
    write8(BNO055_PAGE_ID_ADDR, savePageID);

    // Set the requested operating mode (see section 3.3)
    setMode(modeback);
    delay(20);
  }

  void reset() {
    setMode(OPERATION_MODE_CONFIG);
    write8(BNO055_SYS_TRIGGER_ADDR, 0x20);
    delay(30);
    while (read8(BNO055_CHIP_ID_ADDR) != BNO055_ID) {
        delay(10);
    }
    delay(50);
    // Set to normal power mode
    write8(BNO055_PWR_MODE_ADDR, POWER_MODE_NORMAL);
    delay(10);
  }
}
*/