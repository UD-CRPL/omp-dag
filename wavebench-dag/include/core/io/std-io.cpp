#include "std-io.h"

namespace __core__ {
namespace __io__ {
namespace __std__ {
void close_file(std::ifstream& file) {
    if(file.is_open()) {
        try {
            file.close();
        }
        catch(std::ios_base::failure &exc) {
        	error(true,std::string("Couldn't close the file, the following error was given: ")+exc.what(),RUNTIME_ERROR);
        }
    }
}
void close_file(std::ofstream& file) {
    if(file.is_open()) {
        try {
            file.close();
        }
        catch(std::ios_base::failure &exc) {
        	error(true,std::string("Couldn't close the file, the following error was given: ")+exc.what(),RUNTIME_ERROR);
        }
    }
}
void close_file(std::fstream& file) {
    if(file.is_open()) {
        try {
            file.close();
        }
        catch(std::ios_base::failure &exc) {
        	error(true,std::string("Couldn't close the file, the following error was given: ")+exc.what(),RUNTIME_ERROR);
        }
    }
}

std::vector<std::string>&& read_file(const std::string& file_name) {
    std::ifstream file=open_file(file_name);
    std::vector<std::string> lines;
    if(file.is_open()) {
        std::string line;
        while(std::getline(file,line)) {
            lines.push_back(line);
        }
        close_file(file);
    }
    else {
    	error(true,"Couldn't open the file: "+file_name,RUNTIME_ERROR,throw_error);
    }
    return std::move(lines);
}
}
}
}
