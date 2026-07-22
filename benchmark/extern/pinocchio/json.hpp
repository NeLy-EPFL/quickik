// A minimal recursive-descent JSON reader, just enough to parse this
// benchmark's own fixtures (assets/fixtures.json,
// assets/neuromechfly_ypr_legs.json) -- not a general-purpose JSON library.
// No third-party JSON library is vendored here since none is available
// without a package manager in this environment; the input schema is fixed
// and small enough that a purpose-built reader is simpler than vendoring one.

#pragma once

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct Json {
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool bool_value = false;
  double number_value = 0.0;
  std::string string_value;
  std::vector<Json> array_value;
  std::vector<std::pair<std::string, Json>> object_value;

  bool is_null() const { return type == Type::Null; }
  double as_number() const { return number_value; }
  const std::string &as_string() const { return string_value; }
  const std::vector<Json> &as_array() const { return array_value; }

  const Json &at(const std::string &key) const {
    for (auto &[k, v] : object_value) {
      if (k == key) return v;
    }
    throw std::runtime_error("missing JSON key: " + key);
  }
  const Json &operator[](const std::string &key) const { return at(key); }
  const Json &operator[](size_t i) const { return array_value.at(i); }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string &text) : text_(text) {}

  Json parse() {
    Json v = parse_value();
    return v;
  }

 private:
  const std::string &text_;
  size_t pos_ = 0;

  char peek() { return text_[pos_]; }
  char advance() { return text_[pos_++]; }
  void skip_ws() {
    while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) pos_++;
  }
  void expect(char c) {
    if (advance() != c) throw std::runtime_error("expected '" + std::string(1, c) + "' in JSON");
  }
  bool consume_literal(const char *lit, size_t len) {
    if (text_.compare(pos_, len, lit) == 0) {
      pos_ += len;
      return true;
    }
    return false;
  }

  Json parse_value() {
    skip_ws();
    switch (peek()) {
      case '{':
        return parse_object();
      case '[':
        return parse_array();
      case '"':
        return parse_string();
      case 't': {
        if (!consume_literal("true", 4)) throw std::runtime_error("invalid JSON literal");
        Json v;
        v.type = Json::Type::Bool;
        v.bool_value = true;
        return v;
      }
      case 'f': {
        if (!consume_literal("false", 5)) throw std::runtime_error("invalid JSON literal");
        Json v;
        v.type = Json::Type::Bool;
        v.bool_value = false;
        return v;
      }
      case 'n': {
        if (!consume_literal("null", 4)) throw std::runtime_error("invalid JSON literal");
        return Json{};
      }
      default:
        return parse_number();
    }
  }

  Json parse_object() {
    expect('{');
    Json v;
    v.type = Json::Type::Object;
    skip_ws();
    if (peek() == '}') {
      advance();
      return v;
    }
    while (true) {
      skip_ws();
      Json key = parse_string();
      skip_ws();
      expect(':');
      Json value = parse_value();
      v.object_value.emplace_back(key.string_value, std::move(value));
      skip_ws();
      char c = advance();
      if (c == '}') break;
      if (c != ',') throw std::runtime_error("expected ',' or '}' in JSON object");
    }
    return v;
  }

  Json parse_array() {
    expect('[');
    Json v;
    v.type = Json::Type::Array;
    skip_ws();
    if (peek() == ']') {
      advance();
      return v;
    }
    while (true) {
      v.array_value.push_back(parse_value());
      skip_ws();
      char c = advance();
      if (c == ']') break;
      if (c != ',') throw std::runtime_error("expected ',' or ']' in JSON array");
    }
    return v;
  }

  Json parse_string() {
    expect('"');
    Json v;
    v.type = Json::Type::String;
    std::string &out = v.string_value;
    while (true) {
      char c = advance();
      if (c == '"') break;
      if (c == '\\') {
        char esc = advance();
        switch (esc) {
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'u': pos_ += 4; out += '?'; break;  // not needed for this benchmark's fixtures
          default: throw std::runtime_error("invalid JSON escape");
        }
      } else {
        out += c;
      }
    }
    return v;
  }

  Json parse_number() {
    size_t start = pos_;
    if (peek() == '-') advance();
    while (pos_ < text_.size() && (std::isdigit(static_cast<unsigned char>(peek())) || peek() == '.' || peek() == 'e' ||
                                    peek() == 'E' || peek() == '+' || peek() == '-')) {
      advance();
    }
    Json v;
    v.type = Json::Type::Number;
    v.number_value = std::strtod(text_.c_str() + start, nullptr);
    return v;
  }
};

inline Json parse_json_file(const std::string &path) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("failed to open " + path);
  std::string contents;
  char buf[65536];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) contents.append(buf, n);
  std::fclose(f);
  JsonParser parser(contents);
  return parser.parse();
}
