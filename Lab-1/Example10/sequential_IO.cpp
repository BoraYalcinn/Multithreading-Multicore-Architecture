#include <curl/curl.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

size_t writeCall(void* contents,size_t size,size_t nmemb, std::string* output){
    size_t totalSize = size * nmemb;
    output->append((char*)contents,totalSize);
    return totalSize;
}

int main(){

    CURLcode res = CURLE_OK;
    CURL* curl = curl_easy_init();

    std::vector<std::string> URLs;
    for (int i = 1; i <= 12; i++) {
        URLs.push_back("http://localhost:8001/file" + std::to_string(i) + ".txt");
    }
    std::cout << "-----------SEQUENTIAL-----------" << std::endl ;
    std::vector<std::string> reponses;
    const char* filename = "fetced_data";

    // set timer
    auto start = std::chrono::steady_clock::now();

    for(std::string url :  URLs){
        std::string newResponse;
        

        if(curl){

            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCall);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &newResponse);

            CURLcode res = curl_easy_perform(curl);

            if (res != CURLE_OK) {
                std::cerr << "curl error: " << curl_easy_strerror(res) << std::endl;
            } 
            
        }

        reponses.push_back(newResponse);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeElapsed = end - start;
    std::cout << "Pool size = 12" << " -> Time elapsed = " << timeElapsed.count() << "s" << std::endl;
    std::cout << "-----------------------------" << std::endl ;
    curl_easy_cleanup(curl);  

    return 0;
}