#include <iostream>

int main() {
    int number;
    if (!(std::cin >> number)) {
        std::cerr << "Expected an integer";
        return 1;
    }
    std::cout << number * number << '\n';
    return 0;
}
