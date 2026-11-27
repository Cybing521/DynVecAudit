// Q-BAM M1 engine -- minimal JSON parser implementation.
#include "common/json.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace qbam {

namespace {
std::string escape_json_string_for_dump(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

[[noreturn]] void parse_fail(const std::string& what, size_t at) {
    throw std::runtime_error("JSON parse error at byte " + std::to_string(at) + ": " + what);
}
}  // namespace

void JsonParser::error(const std::string& what) const { parse_fail(what, i); }

void JsonParser::skip_ws() {
    while (i < s.size()) {
        char c = s[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            ++i;
        } else {
            break;
        }
    }
}

JsonValue JsonParser::parse_document() {
    skip_ws();
    JsonValue v = parse_value();
    skip_ws();
    if (i != s.size()) error("trailing characters after document");
    return v;
}

JsonValue JsonParser::parse_value() {
    skip_ws();
    if (i >= s.size()) error("unexpected end of input");
    char c = s[i];
    switch (c) {
        case '{':
            return parse_object();
        case '[':
            return parse_array();
        case '"': {
            JsonValue v;
            v.type_ = JsonValue::Type::String;
            v.str_ = parse_string();
            return v;
        }
        case 't': {
            if (s.compare(i, 4, "true") != 0) error("invalid literal");
            i += 4;
            JsonValue v;
            v.type_ = JsonValue::Type::Bool;
            v.bool_ = true;
            return v;
        }
        case 'f': {
            if (s.compare(i, 5, "false") != 0) error("invalid literal");
            i += 5;
            JsonValue v;
            v.type_ = JsonValue::Type::Bool;
            v.bool_ = false;
            return v;
        }
        case 'n': {
            if (s.compare(i, 4, "null") != 0) error("invalid literal");
            i += 4;
            return JsonValue();
        }
        default:
            if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
            error("unexpected character");
    }
}

JsonValue JsonParser::parse_object() {
    JsonValue v;
    v.type_ = JsonValue::Type::Object;
    ++i;  // '{'
    skip_ws();
    if (i < s.size() && s[i] == '}') {
        ++i;
        return v;
    }
    while (true) {
        skip_ws();
        if (i >= s.size() || s[i] != '"') error("expected object key string");
        std::string key = parse_string();
        skip_ws();
        if (i >= s.size() || s[i] != ':') error("expected ':' after object key");
        ++i;
        JsonValue val = parse_value();
        v.obj_.emplace_back(std::move(key), std::move(val));
        skip_ws();
        if (i >= s.size()) error("unterminated object");
        if (s[i] == ',') {
            ++i;
            continue;
        }
        if (s[i] == '}') {
            ++i;
            return v;
        }
        error("expected ',' or '}' in object");
    }
}

JsonValue JsonParser::parse_array() {
    JsonValue v;
    v.type_ = JsonValue::Type::Array;
    ++i;  // '['
    skip_ws();
    if (i < s.size() && s[i] == ']') {
        ++i;
        return v;
    }
    while (true) {
        v.arr_.push_back(parse_value());
        skip_ws();
        if (i >= s.size()) error("unterminated array");
        if (s[i] == ',') {
            ++i;
            continue;
        }
        if (s[i] == ']') {
            ++i;
            return v;
        }
        error("expected ',' or ']' in array");
    }
}

JsonValue JsonParser::parse_number() {
    size_t start = i;
    if (i < s.size() && s[i] == '-') ++i;
    bool any_digit = false;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
        ++i;
        any_digit = true;
    }
    if (i < s.size() && s[i] == '.') {
        ++i;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
            ++i;
            any_digit = true;
        }
    }
    if (!any_digit) error("invalid number");
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        ++i;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
        bool exp_digit = false;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
            ++i;
            exp_digit = true;
        }
        if (!exp_digit) error("invalid exponent");
    }
    std::string token = s.substr(start, i - start);
    char* endp = nullptr;
    double d = std::strtod(token.c_str(), &endp);
    if (endp == nullptr || *endp != '\0') error("invalid number token");
    JsonValue v;
    v.type_ = JsonValue::Type::Number;
    v.num_ = d;
    return v;
}

std::string JsonParser::parse_string() {
    ++i;  // opening quote
    std::string out;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == '"') {
            ++i;
            return out;
        }
        if (c == '\\') {
            ++i;
            if (i >= s.size()) error("unterminated escape");
            char esc = s[i++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    if (i + 4 > s.size()) error("short \\u escape");
                    unsigned cp = 0;
                    for (int k = 0; k < 4; ++k) {
                        char h = s[i + k];
                        unsigned nib = 0;
                        if (h >= '0' && h <= '9') {
                            nib = static_cast<unsigned>(h - '0');
                        } else if (h >= 'a' && h <= 'f') {
                            nib = static_cast<unsigned>(h - 'a' + 10);
                        } else if (h >= 'A' && h <= 'F') {
                            nib = static_cast<unsigned>(h - 'A' + 10);
                        } else {
                            error("invalid \\u escape");
                        }
                        cp = (cp << 4) | nib;
                    }
                    i += 4;
                    // Encode as UTF-8 (BMP only; surrogate pairs rejected).
                    if (cp >= 0xD800 && cp <= 0xDFFF) error("surrogate escapes unsupported");
                    if (cp < 0x80) {
                        out.push_back(static_cast<char>(cp));
                    } else if (cp < 0x800) {
                        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    } else {
                        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    }
                    break;
                }
                default:
                    error("invalid escape character");
            }
            continue;
        }
        out.push_back(static_cast<char>(c));
        ++i;
    }
    error("unterminated string");
}

JsonValue JsonValue::parse(const std::string& text) {
    JsonParser p(text);
    return p.parse_document();
}

bool JsonValue::as_bool(bool fallback) const { return type_ == Type::Bool ? bool_ : fallback; }

double JsonValue::as_double(double fallback) const {
    if (type_ == Type::Number) return num_;
    if (type_ == Type::Bool) return bool_ ? 1.0 : 0.0;
    return fallback;
}

int64_t JsonValue::as_int(int64_t fallback) const {
    if (type_ == Type::Number) return static_cast<int64_t>(num_);
    return fallback;
}

const std::string& JsonValue::as_string() const {
    if (type_ != Type::String) {
        throw std::runtime_error("JSON value is not a string");
    }
    return str_;
}

const std::vector<JsonValue>& JsonValue::as_array() const {
    if (type_ != Type::Array) {
        throw std::runtime_error("JSON value is not an array");
    }
    return arr_;
}

const std::vector<std::pair<std::string, JsonValue>>& JsonValue::as_object() const {
    if (type_ != Type::Object) {
        throw std::runtime_error("JSON value is not an object");
    }
    return obj_;
}

const JsonValue* JsonValue::find(const std::string& key) const {
    if (type_ != Type::Object) return nullptr;
    for (const auto& kv : obj_) {
        if (kv.first == key) return &kv.second;
    }
    return nullptr;
}

const JsonValue& JsonValue::at(const std::string& key) const {
    const JsonValue* v = find(key);
    if (v == nullptr) {
        throw std::runtime_error("missing required config key: \"" + key + "\"");
    }
    return *v;
}

std::string JsonValue::dump() const {
    switch (type_) {
        case Type::Null: return "null";
        case Type::Bool: return bool_ ? "true" : "false";
        case Type::Number: {
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.12g", num_);
            return std::string(buf);
        }
        case Type::String: return "\"" + escape_json_string_for_dump(str_) + "\"";
        case Type::Array: {
            std::string o = "[";
            for (size_t i = 0; i < arr_.size(); ++i) {
                if (i > 0) o += ',';
                o += arr_[i].dump();
            }
            o += ']';
            return o;
        }
        case Type::Object: {
            std::string o = "{";
            for (size_t i = 0; i < obj_.size(); ++i) {
                if (i > 0) o += ',';
                o += "\"" + escape_json_string_for_dump(obj_[i].first) + "\":";
                o += obj_[i].second.dump();
            }
            o += '}';
            return o;
        }
    }
    return "null";
}

size_t JsonValue::size() const {
    switch (type_) {
        case Type::Array: return arr_.size();
        case Type::Object: return obj_.size();
        case Type::String: return str_.size();
        default: return 0;
    }
}

}  // namespace qbam
