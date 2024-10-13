#include <iostream>
#include <vector> 
#include <cmath>  // 用于 pow 函数
using namespace std;

// 判断一个数是否是水仙花数
bool isPrime(int num) {
    int original = num;
    int sum=0;
    
    // 提取每位数字，计算立方和
    while (num > 0) {
        int digit = num % 10;
        sum += pow(digit, 3);
        num /= 10;
        
    }
    
    // 如果立方和等于原数字，则是水仙花数
    return sum == original;
}

int main() {
    for (int i = 100; i <= 999; ++i) {
        if (isPrime(i)) {
            cout << i << endl;  // 输出水仙花数
        }
    }
    
    return 0;
}
