#include "libunittest/quote.hpp"
#include "libunittest/utilities.hpp"
#include <sstream>


namespace unittest {
namespace core {


std::ostream&
operator<<(std::ostream& os, const unittest::core::quote& q)
{
    os << "\"" << q.message << "\" - " << q.author;
    return os;
}


quote_generator::quote_generator(int seed)
    : generator_((seed <= 0 ? core::now().count() / 1000000 : seed)),
      quotes_{}
{
    quotes_.push_back({"C makes it easy to shoot yourself in the foot; C++ makes it harder, but when you do, it blows away your whole leg.", "Bjarne Stroustrup"});
    quotes_.push_back({"There are only two kinds of languages: the ones people complain about and the ones nobody uses.", "Bjarne Stroustrup"});
    quotes_.push_back({"Within C++, there is a much smaller and cleaner language struggling to get out.", "Bjarne Stroustrup"});
    quotes_.push_back({"In rivers, the water that you touch is the last of what has passed and the first of that which comes; so with present time.", "Leonardo da Vinci"});
    quotes_.push_back({"A person starts to live when he can live outside himself", "Albert Einstein"});
    quotes_.push_back({"You cannot teach a man anything; you can only help him discover it in himself.", "Galileo Galilei"});
    quotes_.push_back({"The Sun, with all the planets revolving around it, and depending on it, can still ripen a bunch of grapes as though it had nothing else in the Universe to do.", "Galileo Galilei"});
    quotes_.push_back({"An investment in knowledge pays the best interest.", "Benjamin Franklin"});
    quotes_.push_back({"An American monkey, after getting drunk on brandy, would never touch it again, and thus is much wiser than most men.", "Charles Darwin"});
    quotes_.push_back({"A true friend is one soul in two bodies.", "Aristotle"});
    quotes_.push_back({"He who is fixed to a star does not change his mind.", "Leonardo da Vinci"});
    quotes_.push_back({"If I have seen further than others, it is by standing upon the shoulders of giants.", "Isaac Newton"});
    quotes_.push_back({"Imagination is more important than knowledge.", "Albert Einstein"});
}

quote
quote_generator::next() const
{
    std::uniform_int_distribution<int> dist(0, quotes_.size() - 1);
    return quotes_[dist(generator_)];
}


} // core
} // unittest
