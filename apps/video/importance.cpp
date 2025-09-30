// gaussian_importance.cpp
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    // Parametri: miu (_t_obj), sigma, pas, nume_fisier_csv
    const double miu   = (argc > 1) ? std::stod(argv[1]) : 0.65;   // center (_t_obj)
    const double sigma = (argc > 2) ? std::stod(argv[2]) : 0.1;   // spread
    const double step  = (argc > 3) ? std::stod(argv[3]) : 0.01;  // increment pentru t
    const std::string out_name = (argc > 4) ? argv[4] : "t_importance.csv";

    if (sigma <= 0.0 || step <= 0.0) {
        std::cerr << "Sigma si step trebuie sa fie > 0.\n";
        return 1;
    }

    std::ofstream out(out_name);
    if (!out) {
        std::cerr << "Nu pot deschide fisierul de iesire: " << out_name << "\n";
        return 1;
    }

    out << "t,importance\n";
    out << std::fixed << std::setprecision(6);

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const double scale = inv_sqrt2 / sigma;

    for (double t = 0.0; t < 1.0; t += step) {
        // (t - miu) / (sigma * sqrt(2))
        const double z = (t - miu) * scale;

        // cdp = 0.5 * (1 + erf(z))
        const double cdp = 0.5 * (1.0 + std::erf(z));

        const double closeness = std::fabs(0.5 - cdp);
        double importance = 1.0 - 2.0 * closeness; // deja in [0,1]
        if (importance < 0.0) importance = 0.0;    // siguranta numerica
        if (importance > 1.0) importance = 1.0;

        out << t << "," << importance << "\n";
    }

    std::cout << "Scris " << out_name << " cu perechile (t,importance).\n";
    return 0;
}
