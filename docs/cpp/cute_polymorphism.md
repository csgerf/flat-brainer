# cute_polymorphism

From the following talk at CppCon 2022:
<br>Lightning Talk: Cute Approach for Polymorphism in C++ - Liad Aben Sour
Asayag - [youtube link](https://www.youtube.com/watch?v=Y0B5AkvBL2w)

## The task

- get a container of many elements (from network, accelerometer, etc.)
- many handlers that can handle many elements
- which handler to use is determined at runtime

```cpp
struct Element {
    // ...
};

struct HandlerA {
    void handle(Element e) {
        // ...
    }
};

struct HandlerB {
    void handle(Element e) {
        // ...
    }
};

```

## Simple Solution: inheritance

```cpp
struct HandlerBase {
    virtual void handle(Element e) = 0;
};

struct HandlerA : HandlerBase {
    void handle(Element e) override {
        // ...
    }
};

struct HandlerB : HandlerBase {
    void handle(Element e) override {
        // ...
    }
};
```

main.cpp

```cpp
std::unique_ptr<HandlerBase> choose_handler(char *argv[]);
std::vector<Element> get_elements();

int main(int argc, char *argv[]) {
    std::unique_ptr<HandlerBase> handler = choose_handler(argv);
    while (true) {
        std::vector<Element> elements = get_elements();
        for (Element  e : elements) {
          h->handle(std::move(e));
        }
    }
    return 0;
}
```

- many virtual calls to small virtual functions
    - bad performance
- Each handler could be written in its own cpp file and main.cpp cpp need only know of HandlerBase
    - good encapsulation
    - fast compile times

## Different approach: Concepts and variants

```cpp
struct HandlerA {
    void handle(Element e) const {
        // ...
    }
};

struct HandlerB {
    void handle(Element e) const{
        // ...
    }
};
```

main.cpp

```cpp

std::variant<HandlerA, HandlerB choose_handler(char *argv[]);
std::vector<Element> get_elements();

int main(int argc, char *argv[]) {
    auto handler = choose_handler(argv);
    while (true) {
        std::vector<Element> elements = get_elements();
        std::visit([& elements](const auto &h) { 
            for (Element  e : elements) {
                h.handle(std::move(e));
            }
        }, handler);
        
    }
    return 0

```

- only one virtual call per vector of elements
    - good performance
- main.cpp needs to know all handlers at compile time
    - no need to know about all the handlers
    - no encapsulation
    - slower compile times

## Revised approach

- only one virtual call per vector of elements
    - good performance
- main.cpp is clean
    - no need to know about all the handlers
    - good encapsulation
    - fast compile times
- easy to add new handlers
    - just add a new handler class
    - no need to modify main.cpp

```cpp
struct HandlerBase {
    virtual void handle(std::span<Element> elements) = 0;
};

template <typename Derived>
struct Handler : HandlerBase {
    void handle(std::span<Element> elements) override {
        for (Element  e : elements) {
            d.handle(std::move(e));
        }
    }
    Derived d;
};

struct HandlerA : Handler<HandlerA> {
    void handle(Element e) {
        // ...
    }
};

struct HandlerB : Handler<HandlerB> {
    void handle(Element e) {
        // ...
    }
};
```

main.cpp

```cpp
std::unique_ptr<HandlerBase> choose_handler(char *argv[]);
std::vector<Element> get_elements();

int main(int argc, char *argv[]) {
    std::unique_ptr<HandlerBase> handler = choose_handler(argv);
    while (true) {
        td::vector<Element> elements = get_elements();
        handler->handle(elements);
    }
    return 0;
}
```