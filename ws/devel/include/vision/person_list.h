// Generated by gencpp from file vision/person_list.msg
// DO NOT EDIT!


#ifndef VISION_MESSAGE_PERSON_LIST_H
#define VISION_MESSAGE_PERSON_LIST_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <vision/person.h>

namespace vision
{
template <class ContainerAllocator>
struct person_list_
{
  typedef person_list_<ContainerAllocator> Type;

  person_list_()
    : list()  {
    }
  person_list_(const ContainerAllocator& _alloc)
    : list(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector< ::vision::person_<ContainerAllocator> , typename std::allocator_traits<ContainerAllocator>::template rebind_alloc< ::vision::person_<ContainerAllocator> >> _list_type;
  _list_type list;





  typedef boost::shared_ptr< ::vision::person_list_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::vision::person_list_<ContainerAllocator> const> ConstPtr;

}; // struct person_list_

typedef ::vision::person_list_<std::allocator<void> > person_list;

typedef boost::shared_ptr< ::vision::person_list > person_listPtr;
typedef boost::shared_ptr< ::vision::person_list const> person_listConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::vision::person_list_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::vision::person_list_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::vision::person_list_<ContainerAllocator1> & lhs, const ::vision::person_list_<ContainerAllocator2> & rhs)
{
  return lhs.list == rhs.list;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::vision::person_list_<ContainerAllocator1> & lhs, const ::vision::person_list_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace vision

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::vision::person_list_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::vision::person_list_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::vision::person_list_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::vision::person_list_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::vision::person_list_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::vision::person_list_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::vision::person_list_<ContainerAllocator> >
{
  static const char* value()
  {
    return "6e0e3b7caba85042fa0a8abdf1c715af";
  }

  static const char* value(const ::vision::person_list_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x6e0e3b7caba85042ULL;
  static const uint64_t static_value2 = 0xfa0a8abdf1c715afULL;
};

template<class ContainerAllocator>
struct DataType< ::vision::person_list_<ContainerAllocator> >
{
  static const char* value()
  {
    return "vision/person_list";
  }

  static const char* value(const ::vision::person_list_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::vision::person_list_<ContainerAllocator> >
{
  static const char* value()
  {
    return "vision/person[] list\n"
"================================================================================\n"
"MSG: vision/person\n"
"string name\n"
"int64 x\n"
"int64 y\n"
;
  }

  static const char* value(const ::vision::person_list_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::vision::person_list_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.list);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct person_list_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::vision::person_list_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::vision::person_list_<ContainerAllocator>& v)
  {
    s << indent << "list[]" << std::endl;
    for (size_t i = 0; i < v.list.size(); ++i)
    {
      s << indent << "  list[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::vision::person_<ContainerAllocator> >::stream(s, indent + "    ", v.list[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // VISION_MESSAGE_PERSON_LIST_H