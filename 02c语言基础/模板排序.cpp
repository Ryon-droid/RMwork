#include <iostream>
#include <vector>
#include <algorithm>  // 用于 std::sort
using namespace std;

int main() {
    int N;
    cin >> N;  // 输入整数 N
    vector<int> numbers(N);  // 用于存储输入的 N 个数
    // 输入 N 个数
    for (int i = 0; i < N; ++i) {
        cin >> numbers[i];
    }

    // 排序
    sort(numbers.begin(), numbers.end());

    // 按照要求输出排序后的数
    for (int i = 0; i < N; ++i) {
        cout << numbers[i];
        if (i >= 0) cout << " ";
    }
    cout << endl;  // 换行

    return 0;
}
