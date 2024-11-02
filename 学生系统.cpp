#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <cstdlib>
using namespace std;

class Student {
private:
	char name[20];
	int id,age;
	float a,b,c,d;
	char ch;
public:
	void input()
	{
		cin.get(name, 21, ',');  //看补充 
		cin>>ch>>age>>ch>>id>>ch>>a>>ch>>b>>ch>>c>>ch>>d;
	};
	float calculate()
	{
		float sum=a+b+c+d;
		return sum/4;
	};
	bool t()
	{
		float sum=(a+b+c+d)/4;
		int a=sum;
		if (a==sum)
			return true;
		else 
			return false;
	}
	void output()
	{
		float mid=calculate();
		cout<<name<<","<<age<<","<<id<<",";
		if (t()==false)
		{
			printf("%.1f",mid);//没有给iomanip头文件，只能选择C语言输出 
		}
		else
			printf("%.0f",mid);
	}
};

int main() {
	Student student;        // 定义类的对象
	student.input();        // 输入数据
	student.calculate();    // 计算平均成绩
	student.output();       // 输出数据
}
