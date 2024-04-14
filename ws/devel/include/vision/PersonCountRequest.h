// Generated by gencpp from file vision/PersonCountRequest.msg
// DO NOT EDIT!


#ifndef VISION_MESSAGE_PERSONCOUNTREQUEST_H
#define VISION_MESSAGE_PERSONCOUNTREQUEST_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace vision
{
template <class ContainerAllocator>
struct PersonCountRequest_
{
  typedef PersonCountRequest_<ContainerAllocator> Type;

  PersonCountRequest_()
    : data()  {
    }
  PersonCountRequest_(const ContainerAllocator& _alloc)
    : data(_alloc)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> _data_type;
  _data_type data;





  typedef boost::shared_ptr< ::vision::PersonCountRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::vision::PersonCountRequest_<ContainerAllocator> const> ConstPtr;

}; // struct PersonCountRequest_

typedef ::vision::PersonCountRequest_<std::allocator<void> > PersonCountRequest;

typedef boost::shared_ptr< ::vision::PersonCountRequest > PersonCountRequestPtr;
typedef boost::shared_ptr< ::vision::PersonCountRequest const> PersonCountRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::vision::PersonCountRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::vision::PersonCountRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::vision::PersonCountRequest_<ContainerAllocator1> & lhs, const ::vision::PersonCountRequest_<ContainerAllocator2> & rhs)
{
  return lhs.data == rhs.data;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::vision::PersonCountRequest_<ContainerAllocator1> & lhs, const ::vision::PersonCountRequest_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace vision

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::vision::PersonCountRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::vision::PersonCountRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::vision::PersonCountRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::vision::PersonCountRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::vision::PersonCountRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::vision::PersonCountRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::vision::PersonCountRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "992ce8a1687cec8c8bd883ec73ca41d1";
  }

  static const char* value(const ::vision::PersonCountRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x992ce8a1687cec8cULL;
  static const uint64_t static_value2 = 0x8bd883ec73ca41d1ULL;
};

template<class ContainerAllocator>
struct DataType< ::vision::PersonCountRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "vision/PersonCountRequest";
  }

  static const char* value(const ::vision::PersonCountRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::vision::PersonCountRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string data\n"
;
  }

  static const char* value(const ::vision::PersonCountRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::vision::PersonCountRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.data);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct PersonCountRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::vision::PersonCountRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::vision::PersonCountRequest_<ContainerAllocator>& v)
  {
    s << indent << "data: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>::stream(s, indent + "  ", v.data);
  }
};

} // namespace message_operations
} // namespace ros

#endif // VISION_MESSAGE_PERSONCOUNTREQUEST_H