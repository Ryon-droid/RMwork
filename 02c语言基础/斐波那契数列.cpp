#include <iostream>
#include <vector>
using namespace std;

// 计算斐波那契数列的第 a 项
int fib(int a) {
    if (a == 1 || a == 2) return 1;
    int p1 = 1, p2 = 1, p3;
    for (int i = 3; i <= a; ++i) {
        p3 = p1 + p2;
        p1 = p2;
        p2 = p3;
    }
    return p3;
}

int main() {
    int n;  
    cin >> n;
    for (int i=1;i<=n;++i){
        int a;
        cin >> a;
        cout<<fib(a)<<endl;
    }
    return 0;
}
