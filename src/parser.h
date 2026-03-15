#ifndef PARSER_H
#define PARSER_H
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
class Parser
{
public:
    std::string _filename;
    Parser(std::string filename);
    ~Parser();
    std::vector<std::vector<double>> read_data();
};

#endif // PARSER_H