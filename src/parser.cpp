#include "parser.h"

Parser::Parser(std::string filename)
{
    _filename = filename;
}

Parser::~Parser()
{
}

std::vector<std::vector<double>> Parser::read_data()
{
    std::vector<std::vector<double>> data;
    std::ifstream file(_filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << _filename << std::endl;
        return data; // Return empty data on failure
    }
    std::string line;
    while (std::getline(file, line))
    {

        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ','))
        {
            try
            {
                row.push_back(std::stod(value)); // Convert string to double
            }
            catch (const std::invalid_argument &e)
            {
                std::cerr << "Invalid number: " << value << " in line: " << line << std::endl;
                // Handle the error as needed, e.g., skip this value or the entire row
            }
        }
        data.push_back(row);
    }
    file.close();
    return data;
}