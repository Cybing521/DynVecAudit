// Q-BAM M1 engine -- minimal JSON parser (config files only; no third-party deps).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace qbam {

class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    JsonValue() = default;

    // Parses a complete JSON document. Throws std::runtime_error on malformed input.
    static JsonValue parse(const std::string& text);

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool as_bool(bool fallback = false) const;
    double as_double(double fallback = 0.0) const;
    int64_t as_int(int64_t fallback = 0) const;
    const std::string& as_string() const;
    const std::vector<JsonValue>& as_array() const;
    const std::vector<std::pair<std::string, JsonValue>>& as_object() const;

    // Object lookup; returns nullptr when absent or when *this is not an object.
    const JsonValue* find(const std::string& key) const;
    const JsonValue& at(const std::string& key) const;

    size_t size() const;
    // Compact JSON serialisation (used for the run-row policy_params summary).
    std::string dump() const;

private:
    friend struct JsonParser;

    Type type_ = Type::Null;
    bool bool_ = false;
    double num_ = 0.0;
    std::string str_;
    std::vector<JsonValue> arr_;
    std::vector<std::pair<std::string, JsonValue>> obj_;
};

struct JsonParser {
    explicit JsonParser(const std::string& text) : s(text) {}
    JsonValue parse_document();
    JsonValue parse_value();
    JsonValue parse_object();
    JsonValue parse_array();
    JsonValue parse_number();
    std::string parse_string();
    void skip_ws();
    [[noreturn]] void error(const std::string& what) const;

    const std::string& s;
    size_t i = 0;
};

}  // namespace qbam
