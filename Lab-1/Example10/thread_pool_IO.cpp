#include <curl/curl.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>

size_t writeCall(void* contents, size_t size, size_t nmemb, std::string* output){
    size_t totalSize = size * nmemb;
    output->append((char*)contents, totalSize);
    return totalSize;
}

void downloadURL(const char* url, std::string& result) {
    CURL* curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCall);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);

        CURLcode res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            std::cerr << "curl error: " << curl_easy_strerror(res) << std::endl;
        }

        curl_easy_cleanup(curl);
    }
}

int main(){

    curl_global_init(CURL_GLOBAL_DEFAULT);

    std::vector<std::string> URLs;
    for (int i = 1; i <= 12; i++) {
        URLs.push_back("http://localhost:8001/file" + std::to_string(i) + ".txt");
        
    }

    std::vector<int> poolSizes = {1, 2, 4, 5, 6, 8, 12};

    std::cout << "-----------THREAD POOL-----------" << std::endl ;

    for (int poolSize : poolSizes) {

        std::vector<std::thread> threads;
        std::vector<std::string> results(URLs.size());

        int urlsPerThread = URLs.size() / poolSize;
        int remainder = URLs.size() % poolSize;

        auto start = std::chrono::steady_clock::now();

        int currentIndex = 0;
        for (int i = 0; i < poolSize; i++) {
            int urlsForThisThread = 0;
            if (remainder != 0) {
                urlsForThisThread = urlsPerThread + 1;
                remainder--;
            } else {
                urlsForThisThread = urlsPerThread;
            }

            int startIdx = currentIndex;
            int endIdx = currentIndex + urlsForThisThread;
            currentIndex = endIdx;

            threads.emplace_back([startIdx, endIdx, &URLs, &results]() {
                for (int j = startIdx; j < endIdx; j++) {
                    downloadURL(URLs[j].c_str(), results[j]);
                }
            });
        }

        for (auto& t : threads){
            t.join();
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> timeElapsed = end - start;

        std::cout << "Pool size = " << poolSize << " -> Time elapsed = " << timeElapsed.count() << "s" << std::endl;
        std::cout << "-----------------------------" << std::endl ;
    }

    curl_global_cleanup();

    return 0;
}