// thermo_driver.c
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/pci.h>
#include <linux/nvidia-smi.h>  // NVIDIA-specific, adapt for AMD/Intel

#define DEVICE_NAME "thermo0"
#define CLASS_NAME  "thermo"

static int major;
static struct class* thermo_class = NULL;
static struct device* thermo_device = NULL;
static struct cdev thermo_cdev;

struct thermo_sample {
    float temp[128];
    float voltage[128];
    uint64_t jitter[128];
    float adc_noise;
    uint64_t timestamp;
} __attribute__((packed));

static struct thermo_sample current_sample;

static int thermo_open(struct inode *inode, struct file *file) {
    return 0;
}

static ssize_t thermo_read(struct file *file, char __user *buf, size_t len, loff_t *off) {
    if (*off > 0) return 0;
    if (len < sizeof(current_sample)) return -EINVAL;

    // Simulated polling (replace with real NVML/ROCM calls)
    for (int i = 0; i < 108; i++) {  // A100 has ~108 SMs
        current_sample.temp[i] = 45.0f + (float)rand()/RAND_MAX * 15.0f;
        current_sample.voltage[i] = 800.0f + (float)rand()/RAND_MAX * 50.0f;
        current_sample.jitter[i] = 10 + rand() % 20;
    }
    current_sample.adc_noise = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    current_sample.timestamp = ktime_get_ns();

    if (copy_to_user(buf, ®t_sample, sizeof(current_sample)))
        return -EFAULT;

    *off += sizeof(current_sample);
    return sizeof(current_sample);
}

static const struct file_operations thermo_fops = {
    .open = thermo_open,
    .read = thermo_read,
};

static int __init thermo_init(void) {
    major = register_chrdev(0, DEVICE_NAME, &thermo_fops);
    if (major < 0) return major;

    thermo_class = class_create(THIS_MODULE, CLASS_NAME);
    thermo_device = device_create(thermo_class, NULL, MKDEV(major, 0), NULL, DEVICE_NAME);

    pr_info("Thermo driver loaded: /dev/%s\n", DEVICE_NAME);
    return 0;
}

static void __exit thermo_exit(void) {
    device_destroy(thermo_class, MKDEV(major, 0));
    class_destroy(thermo_class);
    unregister_chrdev(major, DEVICE_NAME);
}

module_init(thermo_init);
module_exit(thermo_exit);
MODULE_LICENSE("MIT");
